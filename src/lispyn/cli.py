"""Command-line interface for Lispyn.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .compiler import LispynError, check_source, run_hy
from .notebooks import build_notebook, execute_notebook, write_notebook


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lispyn", description="Hy experiments, Python notebooks, PyTorch freedom."
    )
    p.add_argument("--version", action="version", version="lispyn 0.5.0a1")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("build", "Generate a Python Jupyter notebook without executing it"),
        ("execute", "Generate, execute, and save the Jupyter notebook"),
        ("run", "Run the Hy file directly without creating a notebook"),
    ]:
        action = sub.add_parser(name, help=help_text)
        action.add_argument("source", type=Path, help="Trusted .hy source file")
        if name != "run":
            action.add_argument("-o", "--output", type=Path, default=None)
        if name == "execute":
            action.add_argument("--timeout", type=int, default=900, help="Per-cell timeout in seconds")
            action.add_argument("--kernel", default="python3", help="Installed Jupyter kernel name")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        source = check_source(args.source)
        if args.command == "run":
            return run_hy(source)
        output = args.output or source.with_suffix(".ipynb")
        notebook = build_notebook(source)
        if args.command == "build":
            print(f"Generated: {write_notebook(notebook, output)}")
            return 0
        if args.timeout < -1:
            raise LispynError("--timeout must be -1 (unlimited) or a nonnegative number")
        # Save the notebook before running so the generated artifact survives
        # kernel startup errors, and save available outputs after cell errors.
        write_notebook(notebook, output)
        try:
            execute_notebook(notebook, cwd=source.parent, timeout=args.timeout, kernel=args.kernel)
        finally:
            write_notebook(notebook, output)
        print(f"Executed and saved: {Path(output).resolve()}")
        return 0
    except (LispynError, OSError) as exc:
        print(f"lispyn: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # A failed notebook should not be reported as a successful run.
        print(f"lispyn: execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
