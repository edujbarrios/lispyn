"""Portable, offline-verifiable Lispyn experiment capsules.

A capsule is a deterministic ZIP containing the author's original .hy source,
a *completed* standard Jupyter notebook, and optionally aggregate saved-run
JSON. We never execute code, pickle objects, or collect model checkpoints,
datasets, environment variables or local absolute paths automatically.

A SHA-256 manifest detects accidental corruption/changes. It is NOT a digital
signature or authentication of the author. Inspect notebooks before sharing:
their cells and saved output can contain sensitive data.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import tempfile
import zipfile

import nbformat

from . import __version__
from .compiler import LispynError
from .notebook_audit import audit_notebook


_SCHEMA = 1
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_SOURCE = "source/experiment.hy"
_NOTEBOOK = "notebook/experiment.ipynb"
_MANIFEST = "capsule.json"
_ALLOWED = {_SOURCE, _NOTEBOOK, "run/history.json", "run/manifest.json", _MANIFEST}
_MAX_SOURCE = 2 * 1024 * 1024
_MAX_NOTEBOOK = 80 * 1024 * 1024
_MAX_JSON = 4 * 1024 * 1024
_MAX_MANIFEST = 64 * 1024
_LIMITS = {_SOURCE: _MAX_SOURCE, _NOTEBOOK: _MAX_NOTEBOOK,
           "run/history.json": _MAX_JSON, "run/manifest.json": _MAX_JSON,
           _MANIFEST: _MAX_MANIFEST}


def _read_regular(path, label, size_limit):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise LispynError(f"{label} must be an existing non-symlink file")
    if path.stat().st_size > size_limit:
        raise LispynError(f"{label} exceeds the capsule size limit")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise LispynError(f"Cannot read {label}: {exc}") from exc
    if len(content) > size_limit:
        raise LispynError(f"{label} exceeds the capsule size limit")
    return content


def _digest(data):
    return sha256(data).hexdigest()


def _json_document(raw, label):
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise LispynError(f"Invalid {label} JSON") from exc
    if not isinstance(data, dict):
        raise LispynError(f"{label} must be a JSON object")
    return data


def _notebook(raw, source_name, source_hash):
    try:
        notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
        nbformat.validate(notebook)
    except (ValueError, UnicodeError, nbformat.ValidationError) as exc:
        raise LispynError(f"Invalid capsule Jupyter notebook: {exc}") from exc
    data = notebook.metadata.get("lispyn", {})
    if (not isinstance(data, dict) or
            data.get("source") != source_name or
            data.get("source_sha256") != source_hash):
        raise LispynError("Notebook source provenance does not match the archived Hy file")
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code:
        raise LispynError("Capsule notebook has no code cells")
    if any(cell.get("execution_count") is None for cell in code):
        raise LispynError("Capsule notebook has unexecuted code cells")
    if any(out.get("output_type") == "error" for cell in code
           for out in cell.get("outputs", [])):
        raise LispynError("Capsule notebook contains recorded errors")
    return len(code)


def _manifest(files, source_name, code_cells):
    records = [
        {"path": name, "bytes": len(data), "sha256": _digest(data)}
        for name, data in sorted(files.items())
    ]
    return {
        "schema_version": _SCHEMA,
        "generator": "lispyn",
        "lispyn_version": __version__,
        "hy_source_name": source_name,
        "source_sha256": _digest(files[_SOURCE]),
        "executed_code_cells": code_cells,
        "files": records,
    }


def _zip_info(name):
    item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    item.compress_type = zipfile.ZIP_DEFLATED
    item.create_system = 3
    item.external_attr = (stat.S_IFREG | 0o644) << 16
    return item


def create_capsule(source, notebook, output, *, run=None):
    """Create a portable ZIP without executing/recompiling a Hy experiment.

    Strict source-hash and executed-notebook checks prevent accidentally
    publishing stale or incomplete results. Only selected JSON artifacts from
    --run are added; weights, datasets and arbitrary files are not collected.
    Existing ZIP destinations are never overwritten.
    """
    source, notebook, output = map(lambda x: Path(x).expanduser(), (source, notebook, output))
    if source.suffix.lower() != ".hy" or notebook.suffix.lower() != ".ipynb":
        raise LispynError("Capsule needs a .hy source and .ipynb notebook")
    if output.suffix.lower() != ".zip":
        raise LispynError("Capsule output must use a .zip filename")
    if output.exists() or output.is_symlink():
        raise LispynError(f"Refusing to overwrite an existing capsule: {output}")

    files = {
        _SOURCE: _read_regular(source, "Hy source", _MAX_SOURCE),
        _NOTEBOOK: _read_regular(notebook, "Jupyter notebook", _MAX_NOTEBOOK),
    }
    audit = audit_notebook(notebook, source=source)
    if not audit["ready_to_share"] or audit["source_verified"] is not True:
        raise LispynError("Notebook is not fully executed or source provenance does not match")
    cells = _notebook(files[_NOTEBOOK], source.name, _digest(files[_SOURCE]))
    if run is not None:
        directory = Path(run).expanduser()
        if directory.is_symlink() or not directory.is_dir():
            raise LispynError("Run directory must be an existing non-symlink directory")
        history_path = directory / "history.json"
        files["run/history.json"] = _read_regular(
            history_path, "run history.json", _MAX_JSON,
        )
        _json_document(files["run/history.json"], "run history")
        manifest_file = directory / "manifest.json"
        if manifest_file.is_file() or manifest_file.is_symlink():
            files["run/manifest.json"] = _read_regular(
                manifest_file, "run manifest.json", _MAX_JSON,
            )
            manifest = _json_document(files["run/manifest.json"], "run manifest")
            if manifest.get("status") != "completed":
                raise LispynError("Only completed run manifests can be included")
    capsule_manifest = _manifest(files, source.name, cells)
    files[_MANIFEST] = (
        json.dumps(capsule_manifest, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if len(files[_MANIFEST]) > _MAX_MANIFEST:
        raise LispynError("Capsule manifest is too large")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".lispyn-capsule-", suffix=".zip",
            dir=output.parent, delete=False
        ) as scratch:
            temporary = Path(scratch.name)
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, data in sorted(files.items()):
                archive.writestr(_zip_info(name), data)
        if output.exists():
            raise LispynError(f"Refusing to overwrite an existing capsule: {output}")
        temporary.rename(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def verify_capsule(path):
    """Validate ZIP contents and hashes without extracting or executing code.

    Verifies file integrity and Hy source/notebook matching, not authorship or
    the scientific accuracy of measurements.
    """
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".zip":
        raise LispynError("Capsule verification requires an existing .zip file")
    files = {}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            items = archive.infolist()
            names = [item.filename for item in items]
            if len(names) != len(set(names)) or not {_SOURCE, _NOTEBOOK, _MANIFEST} <= set(names):
                raise LispynError("Capsule has duplicate or missing required files")
            if len(names) > len(_ALLOWED) or not set(names) <= _ALLOWED:
                raise LispynError("Capsule contains unrecognized or unsafe entries")
            for info in items:
                name = info.filename
                if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                    raise LispynError("Capsule contains directory or symlink entries")
                if info.file_size > _LIMITS[name]:
                    raise LispynError(f"Capsule member is too large: {name}")
                with archive.open(info) as stream:
                    raw = stream.read(_LIMITS[name] + 1)
                if len(raw) > _LIMITS[name]:
                    raise LispynError(f"Capsule member exceeds size limit: {name}")
                files[name] = raw
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise LispynError(f"Invalid or corrupt capsule: {exc}") from exc

    manifest = _json_document(files[_MANIFEST], "capsule manifest")
    if manifest.get("schema_version") != _SCHEMA or manifest.get("generator") != "lispyn":
        raise LispynError("Unsupported capsule manifest schema or generator")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(files) - 1:
        raise LispynError("Capsule manifest file listing does not match the ZIP")
    names = [row.get("path") for row in entries if isinstance(row, dict)]
    if len(names) != len(entries) or len(set(names)) != len(names) or set(names) != set(files) - {_MANIFEST}:
        raise LispynError("Capsule manifest contains invalid file entries")
    for entry in entries:
        data = files[entry["path"]]
        if (entry.get("sha256") != _digest(data) or
                entry.get("bytes") != len(data)):
            raise LispynError(f"Capsule integrity mismatch: {entry['path']}")
    source_name = manifest.get("hy_source_name")
    if (not isinstance(source_name, str) or
            source_name != Path(source_name).name or
            not source_name.endswith(".hy")):
        raise LispynError("Invalid source filename in capsule manifest")
    if manifest.get("source_sha256") != _digest(files[_SOURCE]):
        raise LispynError("Capsule Hy source SHA-256 mismatch")
    count = _notebook(files[_NOTEBOOK], source_name, _digest(files[_SOURCE]))
    if manifest.get("executed_code_cells") != count:
        raise LispynError("Capsule executed cell count mismatch")
    if "run/history.json" in files:
        _json_document(files["run/history.json"], "run history")
    if "run/manifest.json" in files:
        data = _json_document(files["run/manifest.json"], "run manifest")
        if data.get("status") != "completed":
            raise LispynError("Capsule includes an incomplete run manifest")
    return {
        "valid": True,
        "notebook": _NOTEBOOK,
        "hy_source_name": source_name,
        "source_sha256": manifest["source_sha256"],
        "executed_code_cells": count,
        "run_included": "run/history.json" in files,
        "files_verified": len(entries),
        "integrity": "SHA-256 matched (not an author signature)",
    }
