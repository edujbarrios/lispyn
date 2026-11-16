"""Offline, nonexecuting provenance and execution audit for Jupyter reports.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Only cell counts, output *types*, and Hy source hashes are examined.
No notebook code is run and no cell content or outputs enter the audit report.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re

import nbformat

from .compiler import LispynError


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def audit_notebook(path, *, source=None):
    """Inspect one .ipynb without executing or interpreting its code.

    The optional --source path recomputes the original Hy SHA-256 and checks
    the embedded digest. Without it, provenance is recorded but not
    independently verified. A clean, fully executed notebook with embedded
    provenance is 'ready'; incomplete or error-containing notebooks are not.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".ipynb" or not path.is_file():
        raise LispynError("Notebook audit requires an existing .ipynb file")
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except (OSError, ValueError, nbformat.ValidationError) as exc:
        raise LispynError(f"Cannot audit a valid Jupyter notebook: {exc}") from exc

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    executed = sum(cell.get("execution_count") is not None for cell in code_cells)
    errored = sum(
        any(output.get("output_type") == "error" for output in cell.get("outputs", []))
        for cell in code_cells
    )
    metadata = notebook.metadata.get("lispyn", {})
    source_name = metadata.get("source")
    digest = metadata.get("source_sha256")
    verified = None
    issues = []

    if not code_cells:
        issues.append("no_code_cells")
    if executed != len(code_cells):
        issues.append("unexecuted_code_cells")
    if errored:
        issues.append("execution_errors")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        issues.append("missing_or_invalid_source_sha256")

    if source is not None:
        source = Path(source).expanduser().resolve()
        if source.suffix.lower() != ".hy" or not source.is_file():
            raise LispynError("Source verification requires an existing .hy file")
        actual = sha256(source.read_bytes()).hexdigest()
        verified = isinstance(digest, str) and digest == actual
        if not verified:
            issues.append("source_hash_mismatch")
        if isinstance(source_name, str) and source_name != source.name:
            issues.append("source_filename_mismatch")

    return {
        "notebook": path.name,
        "hy_source": source_name if isinstance(source_name, str) else None,
        "source_sha256": digest if isinstance(digest, str) else None,
        "source_verified": verified,
        "code_cells": len(code_cells),
        "executed_code_cells": executed,
        "unexecuted_code_cells": len(code_cells) - executed,
        "error_cells": errored,
        "ready_to_share": not issues,
        "issues": issues,
    }
