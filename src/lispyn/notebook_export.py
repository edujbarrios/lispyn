"""Export a Lispyn Jupyter experiment to readable, portable Markdown.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Conversion reads notebooks only; it never executes code. HTML, JavaScript and
SVG output are deliberately not copied into the Markdown report.
PNG/JPEG outputs become separate files in a sibling assets directory.
"""
from __future__ import annotations

import base64
import binascii
from pathlib import Path
import re

import nbformat

from .compiler import LispynError

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MAX_IMAGE_BYTES = 10_000_000


def _text(value):
    return "".join(value) if isinstance(value, list) else str(value)


def _fenced(value, language="text"):
    value = _text(value).rstrip("\n")
    longest = max((len(m.group()) for m in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{value}\n{fence}"


def _safe_image(value):
    if not isinstance(value, str):
        value = _text(value)
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise LispynError("Notebook contains an invalid base64 figure") from exc
    if not data or len(data) > _MAX_IMAGE_BYTES:
        raise LispynError("Notebook image exceeds the supported 10 MB size or is empty")
    return data


def export_markdown(source, output, *, include_code=True):
    """Export Markdown, code and safe visible outputs from an .ipynb.

    Text/Markdown output and PNG/JPEG figures are included; notebook-only
    HTML/JS/SVG output is intentionally omitted. No cell is executed.
    Refuse to overwrite output or asset files and preserve Hy source SHA.
    Returns the Markdown path.
    """
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source.suffix.lower() != ".ipynb" or not source.is_file():
        raise LispynError("Markdown export requires an existing .ipynb file")
    if output.suffix.lower() != ".md":
        raise LispynError("Markdown export output must end in .md")
    if not isinstance(include_code, bool):
        raise TypeError("include_code must be Boolean")
    if source == output or output.exists():
        raise LispynError(f"Refusing to overwrite existing output: {output}")
    try:
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
    except (OSError, ValueError, nbformat.ValidationError) as exc:
        raise LispynError(f"Cannot read a valid Jupyter notebook: {exc}") from exc

    metadata = notebook.metadata.get("lispyn", {})
    name = metadata.get("source", source.stem)
    digest = metadata.get("source_sha256")
    lines = [f"# {source.stem} — experiment report", "",
             "Generated from a Jupyter notebook by Lispyn. "
             "This export does not execute notebook cells.", "",
             f"**Hy source:** `{name}`"]
    if digest:
        lines.append(f"**Hy source SHA-256:** `{digest}`")
    lines += ["", "---", ""]

    figures = []
    for cell in notebook.cells:
        kind = cell.cell_type
        if kind == "markdown":
            lines += [cell.source.rstrip(), ""]
        elif kind == "raw":
            lines += [_fenced(cell.source), ""]
        elif kind == "code":
            if include_code and cell.source.strip():
                lines += [_fenced(cell.source, "python"), ""]
            for item in cell.get("outputs", []):
                kind = item.get("output_type")
                if kind == "stream":
                    value = _ANSI.sub("", _text(item.get("text", "")))
                    if value.strip():
                        lines += [_fenced(value), ""]
                elif kind == "error":
                    traceback = item.get("traceback") or [
                        f"{item.get('ename', 'Error')}: {item.get('evalue', '')}"
                    ]
                    lines += [_fenced(_ANSI.sub("", _text(traceback)), "text"), ""]
                elif kind in ("execute_result", "display_data"):
                    data = item.get("data", {})
                    for mime, extension in (("image/png", "png"), ("image/jpeg", "jpg")):
                        if mime in data:
                            binary = _safe_image(data[mime])
                            filename = f"figure-{len(figures) + 1:03d}.{extension}"
                            figures.append((filename, binary))
                            lines += [
                                f"![Notebook figure {len(figures)}]"
                                f"({output.stem}_assets/{filename})", "",
                            ]
                            break
                    else:
                        if "text/markdown" in data:
                            lines += [_text(data["text/markdown"]), ""]
                        elif "text/plain" in data:
                            lines += [_fenced(data["text/plain"]), ""]
                        # Deliberately do not publish raw HTML, SVG or JavaScript.

    assets = output.parent / f"{output.stem}_assets"
    if figures and assets.exists():
        raise LispynError(f"Refusing to overwrite existing figure assets: {assets}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if figures:
        assets.mkdir()
        for filename, data in figures:
            (assets / filename).write_bytes(data)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
