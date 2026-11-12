"""Generate and optionally execute Python notebooks from trusted Hy programs.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path
import re
import tempfile

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .compiler import LispynError, check_source, compile_hy


ATTRIBUTION = (
    "Generated with **Lispyn** — originally created by **Eduardo J. Barrios**.\n\n"
    "Source: [Lispyn](https://github.com/edujbarrios/lispyn) "
    "(Apache-2.0)."
)

# Jupytext-inspired markers are comments, so the same .hy file can still run.
CELL_MARKER = re.compile(r"^\s*;\s*%%(?:\s*\[(code|markdown)\])?\s*$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Partition a Hy file using '; %% [code|markdown]' comments.

    Markdown lines must be Hy comments; this keeps the .hy source directly
    executable. Code chunks are compiled independently in notebook mode.
    """
    sections: list[tuple[str, str]] = []
    kind = "code"
    lines: list[str] = []

    def flush() -> None:
        if not any(line.strip() for line in lines):
            return
        if kind == "markdown":
            result: list[str] = []
            for line in lines:
                if not line.strip():
                    result.append("")
                    continue
                match = re.match(r"^\s*; ?(.*)$", line.rstrip("\r\n"))
                if match is None:
                    raise LispynError(
                        "Markdown sections must use '; ' comment lines "
                        "so the Hy source stays executable"
                    )
                result.append(match.group(1))
            content = "\n".join(result).strip()
        else:
            content = "".join(lines).strip()
        if content:
            sections.append((kind, content))

    for line in text.splitlines(keepends=True):
        match = CELL_MARKER.match(line.strip("\r\n"))
        if match:
            flush()
            lines = []
            kind = match.group(1) or "code"
        else:
            lines.append(line)
    flush()
    return sections


def compile_section(code: str, *, source: Path) -> str:
    """Compile a code cell without modifying the user's original file."""
    with tempfile.TemporaryDirectory(prefix="lispyn-cell-") as directory:
        temp_source = Path(directory) / source.name
        temp_source.write_text(code + "\n", encoding="utf-8")
        return compile_hy(temp_source, cwd=source.parent)


def build_notebook(source: Path) -> nbformat.NotebookNode:
    """Compile Hy source to Python notebook cells without executing them.

    Existing files without section markers remain a single translated cell.
    Each explicitly separated code section is compiled independently, so a
    macro used in a section must be required in that same section. Runtime
    variables persist between executed notebook cells.
    """
    source = check_source(source)
    text = source.read_text(encoding="utf-8")
    parts = split_sections(text)
    has_markers = any(CELL_MARKER.match(line) for line in text.splitlines())
    cells = [new_markdown_cell(f"# {source.stem}\n\n{ATTRIBUTION}")]

    if not has_markers:
        # Preserve original compilation behavior, path and error messages.
        cells.append(new_code_cell(compile_hy(source)))
    else:
        for kind, content in parts:
            if kind == "markdown":
                cells.append(new_markdown_cell(content))
            else:
                cells.append(new_code_cell(compile_section(content, source=source)))

    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "lispyn": {"source": source.name, "author": "Eduardo J. Barrios"},
        },
    )


def write_notebook(notebook: nbformat.NotebookNode, output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, output)
    return output


def execute_notebook(notebook: nbformat.NotebookNode, *, cwd: Path, timeout: int = 900,
                     kernel: str = "python3") -> nbformat.NotebookNode:
    """Run Python cells in a Jupyter kernel, without spawning a web UI."""
    try:
        from nbclient import NotebookClient
    except ImportError as exc:
        raise LispynError(
            'Notebook execution requires extras: pip install -e ".[notebook]"'
        ) from exc
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel,
        resources={"metadata": {"path": str(cwd.resolve())}},
    )
    return client.execute()
