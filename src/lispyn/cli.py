"""Command-line interface for Lispyn.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compiler import LispynError, check_source, run_hy
from .notebooks import build_notebook, execute_notebook, write_notebook
from .reproducibility import summarize_run
from .scaffold import create_experiment
from .gallery import generate_gallery
from .doctor import inspect_environment, format_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lispyn", description="Hy experiments, Python notebooks, PyTorch freedom."
    )
    p.add_argument("--version", action="version", version="lispyn 0.14.0a1")
    sub = p.add_subparsers(dest="command", required=True)
    health = sub.add_parser("doctor", help="Inspect installation and optional tools without modifying anything")
    health.add_argument("--json", action="store_true", help="Print machine-readable environment summary")
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
    gallery = sub.add_parser(
        "gallery", help="Compile examples into example_generated_notebooks/ without execution"
    )
    gallery.add_argument("--source-dir", type=Path, default=Path("examples"))
    gallery.add_argument("--output-dir", type=Path, default=Path("example_generated_notebooks"))
    starter = sub.add_parser("new", help="Create a ready-to-edit Hy/Jupyter experiment")
    starter.add_argument("source", type=Path, help="Path for a new .hy file")
    report = sub.add_parser("report", help="Summarize a saved Lispyn experiment")
    report.add_argument("directory", type=Path, help="Directory with manifest.json")
    report.add_argument("--plot", type=Path, help="Save an optional PNG/SVG metric plot")
    report.add_argument("--metric", default="loss", help="Metric to plot (default: loss)")
    report.add_argument("--theme", default="clean", choices=["clean", "dark", "paper"])
    report.add_argument("--title", help="Custom plot title")
    report.add_argument("--xlabel", default="Epoch", help="Horizontal axis label")
    report.add_argument("--ylabel", help="Vertical axis label")
    report.add_argument("--train-color", help="Matplotlib training-curve color")
    report.add_argument("--val-color", help="Matplotlib validation-curve color")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = inspect_environment()
            print(json.dumps(report, indent=2) if args.json else format_report(report))
            return 0
        if args.command == "gallery":
            paths = generate_gallery(args.source_dir, args.output_dir)
            for path in paths:
                print(f"Generated: {path}")
            return 0
        if args.command == "new":
            print(f"Created: {create_experiment(args.source)}")
            return 0
        if args.command == "report":
            print(summarize_run(args.directory))
            if args.plot is not None:
                from .plots import plot_history
                fig = plot_history(args.directory / "history.json",
                                   metric=args.metric, save=args.plot,
                                   theme=args.theme, title=args.title,
                                   xlabel=args.xlabel, ylabel=args.ylabel,
                                   train_color=args.train_color,
                                   val_color=args.val_color)
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"Plot saved: {args.plot.expanduser().resolve()}")
            return 0
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
