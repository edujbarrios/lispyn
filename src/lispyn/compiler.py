"""Compile a trusted Hy file into Python using Hy's official CLI.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


class LispynError(Exception):
    """An expected error while building or running a Lispyn experiment."""


def check_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise LispynError(f"File not found: {source}")
    if source.suffix != ".hy":
        raise LispynError("Input must be a .hy file")
    return source


def compile_hy(source: Path, *, cwd: Path | None = None) -> str:
    """Translate a Hy source file to Python; never execute its output here.

    IMPORTANT: Hy macros may run arbitrary Python at *compile time*. Only build
    code you trust. This is not a sandbox.
    """
    source = check_source(source)
    hy2py = shutil.which("hy2py")
    if hy2py is None:
        raise LispynError("hy2py not found. Install Lispyn with: pip install -e .")
    result = subprocess.run(
        [hy2py, str(source)],
        cwd=cwd or source.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise LispynError(
            "Hy compilation failed:\n" + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout


def run_hy(source: Path) -> int:
    """Execute trusted Hy source without creating a notebook."""
    source = check_source(source)
    hy = shutil.which("hy")
    if hy is None:
        raise LispynError("hy not found. Install Lispyn with: pip install -e .")
    return subprocess.run([hy, str(source)], cwd=source.parent, check=False).returncode
