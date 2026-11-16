"""Shareable, deterministic Hy/Jupyter experiment capsules and integrity checks."""

from hashlib import sha256
import json
from pathlib import Path
import stat
import zipfile

from nbformat import write
from nbformat.v4 import new_code_cell, new_notebook, new_output
import pytest

from lispyn.capsule import create_capsule, verify_capsule
from lispyn.cli import main
from lispyn.compiler import LispynError


def experiment(tmp_path, *, executed=True, error=False, matching=True):
    source = tmp_path / "study.hy"
    source.write_text('(print "complete")\n', encoding="utf-8")
    output = new_output("stream", name="stdout", text="complete\n")
    if error:
        output = new_output("error", ename="RuntimeError", evalue="failed",
                            traceback=["RuntimeError: failed"])
    notebook = new_notebook(
        cells=[new_code_cell("print('complete')",
                             execution_count=1 if executed else None,
                             outputs=[output] if executed else [])],
        metadata={"lispyn": {
            "source": "study.hy",
            "source_sha256": sha256(
                source.read_bytes() if matching else b"old source"
            ).hexdigest(),
        }},
    )
    path = tmp_path / "study.ipynb"
    write(notebook, path)
    return source, path


def test_capsule_roundtrip_is_deterministic_and_contains_only_expected_files(tmp_path):
    source, notebook = experiment(tmp_path)
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    assert create_capsule(source, notebook, a) == a.resolve()
    assert create_capsule(source, notebook, b) == b.resolve()
    assert a.read_bytes() == b.read_bytes()
    report = verify_capsule(a)
    assert report["valid"] is True
    assert report["hy_source_name"] == "study.hy"
    assert report["executed_code_cells"] == 1
    assert report["run_included"] is False
    assert report["files_verified"] == 2
    assert report["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(a) as zipped:
        assert zipped.namelist() == [
            "capsule.json", "notebook/experiment.ipynb", "source/experiment.hy"
        ]
        manifest = json.loads(zipped.read("capsule.json"))
        assert manifest["schema_version"] == 1
        assert manifest["source_sha256"] == report["source_sha256"]
        assert len(manifest["files"]) == 2
        assert all("sha256" in file for file in manifest["files"])
        assert all(file.date_time == (1980, 1, 1, 0, 0, 0)
                   for file in zipped.infolist())


def test_capsule_optional_run_only_includes_scalar_json_not_checkpoints(tmp_path):
    source, notebook = experiment(tmp_path)
    run = tmp_path / "runs" / "baseline"
    run.mkdir(parents=True)
    (run / "history.json").write_text(
        '{"train_loss": [1.1, 0.7], "val_loss": [1.2, 0.8]}\n'
    )
    (run / "manifest.json").write_text(
        '{"status": "completed", "model": "native PyTorch"}\n'
    )
    (run / "best.pt").write_bytes(b"private model weights")
    (run / "dataset.csv").write_text("sensitive,raw,data")
    path = tmp_path / "experiment.zip"
    create_capsule(source, notebook, path, run=run)
    info = verify_capsule(path)
    assert info["run_included"] is True
    assert info["files_verified"] == 4
    with zipfile.ZipFile(path) as z:
        assert "run/history.json" in z.namelist()
        assert "run/manifest.json" in z.namelist()
        assert not any("best.pt" in n or "dataset" in n for n in z.namelist())
        assert b"private model weights" not in path.read_bytes()


def test_capsule_cli_create_and_verify_json(tmp_path, capsys):
    source, notebook = experiment(tmp_path)
    archive = tmp_path / "study.zip"
    assert main(["capsule", str(source), str(notebook), "-o", str(archive)]) == 0
    assert "Review notebook cells" in capsys.readouterr().out
    assert main(["verify-capsule", str(archive), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is True
    assert "not an author signature" in data["integrity"]
    assert main(["verify-capsule", str(archive)]) == 0
    assert "Capsule verified" in capsys.readouterr().out
    assert main(["capsule", str(source), str(notebook), "-o", str(archive)]) == 1
    assert "overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("executed,error,matching,diagnostic", [
    (False, False, True, "not fully executed"),
    (True, True, True, "not fully executed"),
    (True, False, False, "provenance does not match"),
])
def test_capsule_rejects_incomplete_failed_or_stale_notebooks(
    tmp_path, executed, error, matching, diagnostic
):
    source, notebook = experiment(
        tmp_path, executed=executed, error=error, matching=matching
    )
    path = tmp_path / "study.zip"
    with pytest.raises(LispynError, match=diagnostic):
        create_capsule(source, notebook, path)
    assert not path.exists()


def test_capsule_refuses_invalid_paths_and_outputs(tmp_path):
    source, notebook = experiment(tmp_path)
    with pytest.raises(LispynError, match="zip filename"):
        create_capsule(source, notebook, tmp_path / "study.tar")
    with pytest.raises(LispynError, match="non-symlink"):
        create_capsule(source, tmp_path / "missing.ipynb", tmp_path / "out.zip")
    link = tmp_path / "link.hy"
    link.symlink_to(source)
    with pytest.raises(LispynError, match="non-symlink"):
        create_capsule(link, notebook, tmp_path / "out.zip")
    with pytest.raises(LispynError, match="existing .zip"):
        verify_capsule(tmp_path / "missing.zip")


def test_capsule_rejects_incomplete_optional_run(tmp_path):
    source, notebook = experiment(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.json").write_text('{"val_loss": [0.5]}')
    (run / "manifest.json").write_text('{"status": "failed"}')
    with pytest.raises(LispynError, match="Only completed"):
        create_capsule(source, notebook, tmp_path / "study.zip", run=run)


def _rewrite_capsule(original, output, modify, *, extra=None):
    with zipfile.ZipFile(original) as archive:
        members = {entry.filename: archive.read(entry.filename)
                   for entry in archive.infolist()}
    modify(members)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if extra:
            archive.writestr(*extra)


def test_verifier_detects_tampering_without_extracting_or_executing(tmp_path):
    source, notebook = experiment(tmp_path)
    original = tmp_path / "study.zip"
    create_capsule(source, notebook, original)
    changed = tmp_path / "modified.zip"
    _rewrite_capsule(
        original, changed,
        lambda m: m.__setitem__(
            "source/experiment.hy", b'(print "modified")\n'
        )
    )
    with pytest.raises(LispynError, match="integrity mismatch"):
        verify_capsule(changed)
    # Verify does not write files in the working directory or run embedded code.
    assert not (tmp_path / "modified.hy").exists()


def test_verifier_rejects_unsafe_and_duplicate_members(tmp_path):
    source, notebook = experiment(tmp_path)
    original = tmp_path / "study.zip"
    create_capsule(source, notebook, original)
    traversal = tmp_path / "traversal.zip"
    _rewrite_capsule(
        original, traversal, lambda m: None,
        extra=("../outside.txt", b"unsafe")
    )
    with pytest.raises(LispynError, match="unsafe entries"):
        verify_capsule(traversal)

    duplicated = tmp_path / "duplicate.zip"
    _rewrite_capsule(
        original, duplicated, lambda m: None,
        extra=("source/experiment.hy", b"duplicate")
    )
    with pytest.raises(LispynError, match="duplicate"):
        verify_capsule(duplicated)


def test_verifier_rejects_bad_manifest_and_notebook_provenance(tmp_path):
    source, notebook = experiment(tmp_path)
    original = tmp_path / "study.zip"
    create_capsule(source, notebook, original)
    bad_manifest = tmp_path / "bad-manifest.zip"
    _rewrite_capsule(
        original, bad_manifest,
        lambda m: m.__setitem__("capsule.json", b'{"schema_version": 999}'),
    )
    with pytest.raises(LispynError, match="schema"):
        verify_capsule(bad_manifest)

    # A forged manifest can be internally consistent, but the notebook still
    # has to point at the actual original Hy source hash.
    forged = tmp_path / "forged.zip"
    def modify(members):
        modified = b'(print "forged")\n'
        members["source/experiment.hy"] = modified
        manifest = json.loads(members["capsule.json"])
        manifest["source_sha256"] = sha256(modified).hexdigest()
        for entry in manifest["files"]:
            if entry["path"] == "source/experiment.hy":
                entry["sha256"] = sha256(modified).hexdigest()
                entry["bytes"] = len(modified)
        members["capsule.json"] = json.dumps(manifest).encode()
    _rewrite_capsule(original, forged, modify)
    with pytest.raises(LispynError, match="provenance"):
        verify_capsule(forged)


def test_no_overwrite_and_rejects_invalid_archive(tmp_path):
    source, notebook = experiment(tmp_path)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"sentinel")
    with pytest.raises(LispynError, match="overwrite"):
        create_capsule(source, notebook, archive)
    assert archive.read_bytes() == b"sentinel"
    with pytest.raises(LispynError, match="Invalid or corrupt"):
        verify_capsule(archive)


def test_citation_is_discoverable_and_version_aligned():
    import lispyn
    path = Path(__file__).resolve().parents[1] / "CITATION.cff"
    text = path.read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in text
    assert "family-names: Barrios" in text
    assert "given-names: Eduardo J." in text
    assert "repository-code:" in text
    assert "license: Apache-2.0" in text
    assert f"version: {lispyn.__version__}" in text
