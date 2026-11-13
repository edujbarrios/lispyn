"""Generate and execute Python Jupyter notebooks from trusted Hy programs.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from hashlib import sha256
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

# Comment markers work in both ordinary Hy execution and notebook compilation.
CELL_MARKER = re.compile(r"^\s*;\s*%%(?:\s*\[(code|markdown)\])?\s*$")
# A literal top-level markdown macro is a first-class cell boundary.
MARKDOWN_FORM = re.compile(r"(?m)^\s*\((?:md|markdown)(?:\s|\))")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Partition a Hy file using '; %% [code|markdown]' comments."""
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


def split_markdown_macros(code: str) -> list[tuple[str, str]]:
    """Extract literal, standalone top-level (md "...") forms as Markdown.

    Hy's reader supplies form positions; we never evaluate Markdown arguments
    or use regex to guess where a nested or multiline Lisp form ends.
    The directive must be the only form on its start/end lines.
    """
    if not MARKDOWN_FORM.search(code):
        return [("code", code)] if code.strip() else []
    import hy
    from hy.models import Expression, String

    try:
        forms = list(hy.read_many(code))
    except Exception as exc:
        raise LispynError(f"Cannot parse Hy source for Markdown cells: {exc}") from exc

    lines = code.splitlines(keepends=True)
    sections = []
    cursor = 0
    found = False

    def append_code(end):
        chunk = "".join(lines[cursor:end]).strip()
        if chunk and any(line.strip() and not line.lstrip().startswith(";")
                         for line in chunk.splitlines()):
            sections.append(("code", chunk))

    for index, form in enumerate(forms):
        if not (isinstance(form, Expression) and form and
                str(form[0]) in ("md", "markdown")):
            continue
        start = form.start_line - 1
        end = form.end_line
        if (start < cursor or
                (index > 0 and forms[index - 1].end_line >= form.start_line) or
                (index + 1 < len(forms) and
                 forms[index + 1].start_line <= form.end_line)):
            raise LispynError("Markdown macros must occupy their own lines")
        if not form[1:] or not all(isinstance(part, String) for part in form[1:]):
            raise LispynError("(md ...) accepts one or more literal Markdown strings only")
        content = "\n\n".join(str(part) for part in form[1:]).strip()
        if not content:
            raise LispynError("Markdown cell text cannot be empty")
        append_code(start)
        sections.append(("markdown", content))
        cursor = end
        found = True

    if not found:
        raise LispynError("Markdown macros must be top-level forms")
    append_code(len(lines))
    return sections


def compile_section(code: str, *, source: Path) -> str:
    """Compile a code cell without modifying the user's source file."""
    with tempfile.TemporaryDirectory(prefix="lispyn-cell-") as directory:
        temp_source = Path(directory) / source.name
        temp_source.write_text(code + "\n", encoding="utf-8")
        return compile_hy(temp_source, cwd=source.parent)


def build_notebook(source: Path) -> nbformat.NotebookNode:
    """Generate notebook cells without executing Python.

    Hy Markdown forms and the old '; %%' comment markers are supported.
    Each code cell is compiled independently; repeat (require ...) for any
    macro you use in a later code cell.
    """
    source = check_source(source)
    text = source.read_text(encoding="utf-8")
    parts = []
    for kind, content in split_sections(text):
        if kind == "code":
            parts.extend(split_markdown_macros(content))
        else:
            parts.append((kind, content))

    has_markers = any(CELL_MARKER.match(line) for line in text.splitlines())
    has_markdown_forms = any(MARKDOWN_FORM.search(content) for kind, content in
                             split_sections(text) if kind == "code")
    cells = [new_markdown_cell(f"# {source.stem}\n\n{ATTRIBUTION}")]

    if not has_markers and not has_markdown_forms:
        cells.append(new_code_cell(compile_hy(source)))
    else:
        for kind, content in parts:
            if kind == "markdown":
                cells.append(new_markdown_cell(content))
            else:
                cells.append(new_code_cell(compile_section(content, source=source)))

    # Stable IDs mean gallery generation does not create meaningless git diffs.
    for index, cell in enumerate(cells):
        digest = sha256(f"{source.name}:{index}:{cell.cell_type}:{cell.source}".encode()).hexdigest()
        cell.id = "lispyn-" + digest[:16]

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
