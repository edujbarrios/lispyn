"""Generate and optionally execute Jupyter notebooks from trusted Hy programs.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .compiler import LispynError, check_source, compile_hy


ATTRIBUTION = (
    "Generated with **Lispyn** — originally created by **Eduardo J. Barrios**.\n\n"
    "Source: [Lispyn](https://github.com/edujbarrios/lispyn) "
    "(Apache-2.0)."
)


def build_notebook(source: Path) -> nbformat.NotebookNode:
    """Create a notebook in memory without executing Python cells.

    v0.1 keeps the complete translated program in one cell so Hy macros and
    Python definitions retain their shared compilation and execution context.
    """
    source = check_source(source)
    code = compile_hy(source)
    return new_notebook(
        cells=[
            new_markdown_cell(f"# {source.stem}\n\n{ATTRIBUTION}"),
            new_code_cell(code),
        ],
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
    """Run notebook cells in an existing Python Jupyter kernel.

    The caller is responsible for saving the notebook even on cell failure.
    """
    try:
        from nbclient import NotebookClient
    except ImportError as exc:
        raise LispynError(
            'Notebook execution requires extra dependencies: pip install -e ".[notebook]"'
        ) from exc
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel,
        resources={"metadata": {"path": str(cwd.resolve())}},
    )
    return client.execute()
