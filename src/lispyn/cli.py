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
    p.add_argument("--version", action="version", version="lispyn 0.22.0a1")
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
    compare = sub.add_parser("compare", help="Compare two or more saved Lispyn runs")
    compare.add_argument("directories", nargs="+", type=Path, help="Run directories with history.json")
    compare.add_argument("--metric", default="loss", help="Metric name without train_/val_ prefix")
    compare.add_argument("--split", choices=("train", "val"), default="val")
    compare.add_argument("--mode", choices=("min", "max"), default=None)
    compare.add_argument("--baseline", help="Run directory name for relative best/final gains")
    compare.add_argument("--output", type=Path, help="Export summary as .md, .csv, or .json")
    compare.add_argument("--plot", type=Path, help="Save combined PNG/SVG plot (requires plots extra)")
    compare.add_argument("--theme", default="docusaurus-dark",
                         choices=("clean", "paper", "dark", "docusaurus-dark"))
    compare.add_argument("--palette", default="turquoise-spectrum",
                         choices=("turquoise-spectrum",))
    cv = sub.add_parser("cv-summary", help="Summarize numeric cross-validation fold scores")
    cv.add_argument("scores", nargs="+", type=float, help="One metric score per validation fold")
    cv.add_argument("--metric", default="score", help="Metric name, e.g. val_accuracy")
    cv.add_argument("--output", type=Path, help="Optional .json summary output")
    starter = sub.add_parser("new", help="Create a ready-to-edit Hy/Jupyter experiment")
    starter.add_argument("source", type=Path, help="Path for a new .hy file")
    report = sub.add_parser("report", help="Summarize a saved Lispyn experiment")
    report.add_argument("directory", type=Path, help="Directory with manifest.json")
    report.add_argument("--plot", type=Path, help="Save an optional PNG/SVG metric plot")
    report.add_argument("--metric", default="loss", help="Metric to plot (default: loss)")
    report.add_argument("--theme", default="clean", choices=["clean", "dark", "paper", "docusaurus-dark"])
    report.add_argument("--title", help="Custom plot title")
    report.add_argument("--subtitle", help="Optional secondary title")
    report.add_argument("--palette", choices=["turquoise-spectrum"],
                        help="Named color palette for the curves")
    report.add_argument("--linewidth", type=float, default=2.2)
    report.add_argument("--fill-alpha", type=float, default=0.0)
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
        if args.command == "compare":
            from .comparison import compare_experiments, format_comparison, save_comparison
            summary = compare_experiments(args.directories, metric=args.metric,
                                          split=args.split, mode=args.mode, baseline=args.baseline)
            print(format_comparison(summary))
            if args.output is not None:
                print(f"Summary saved: {save_comparison(summary, args.output)}")
            if args.plot is not None:
                from .plots import plot_comparison
                import matplotlib.pyplot as plt
                sources = {path.name: path / "history.json" for path in args.directories}
                fig = plot_comparison(sources, metric=args.metric, split=args.split,
                                      theme=args.theme, palette=args.palette,
                                      save=args.plot)
                plt.close(fig)
                print(f"Plot saved: {args.plot.expanduser().resolve()}")
            return 0
        if args.command == "cv-summary":
            from .cross_validation import summarize_cv_scores
            summary = summarize_cv_scores(args.scores, metric=args.metric)
            report = json.dumps(summary, indent=2)
            print(report)
            if args.output is not None:
                path = args.output.expanduser()
                if path.suffix.lower() != ".json":
                    raise LispynError("CV summary output must be .json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(report + "\n", encoding="utf-8")
                print(f"Summary saved: {path}")
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
                                    theme=args.theme, title=args.title, subtitle=args.subtitle,
                                   palette=args.palette, linewidth=args.linewidth,
                                   fill_alpha=args.fill_alpha,
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
