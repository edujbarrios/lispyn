"""Reproducible, privacy-minimal leaderboards from saved Lispyn runs.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

This module reads history.json and the optional completion status in
manifest.json. It never loads model checkpoints, datasets, or arbitrary code.
"""

from __future__ import annotations

import csv
from itertools import islice
import json
import math
from pathlib import Path

from .plots import _read_history, _values
from .reporting import _metric_direction


_MAX_CANDIDATES = 500


def _valid_metric(metric, split, mode, rank_by, top, strict):
    if not isinstance(metric, str) or not metric.strip() or metric.startswith(("val_", "train_")):
        raise ValueError("metric must be an unprefixed nonempty name such as 'loss'")
    if split not in ("val", "train"):
        raise ValueError("split must be 'val' or 'train'")
    if mode not in (None, "min", "max"):
        raise ValueError("mode must be 'min', 'max' or None")
    if rank_by not in ("best", "final"):
        raise ValueError("rank_by must be 'best' or 'final'")
    if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 100:
        raise ValueError("top must be an integer between 1 and 100")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a Boolean")


def _error(skipped, name, reason, *, strict):
    if strict:
        raise ValueError(f"Run {name!r}: {reason}")
    skipped.append({"run": name, "reason": reason})


def discover_leaderboard(root, *, metric="loss", split="val", mode=None,
                         rank_by="best", top=10, strict=False):
    """Rank saved experiment histories under ROOT, recursively.

    Completed native Lispyn runs and legacy history-only directories can be
    ranked; failed/in-progress manifests are skipped. Invalid/absent metric
    histories are skipped by default or fail with strict=True. All paths
    shown in the report are relative to root; models and data are not read.
    A maximum of 500 candidates prevents accidental huge scans.
    """
    _valid_metric(metric, split, mode, rank_by, top, strict)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Experiments root is not a directory: {root}")
    candidates = list(islice(root.rglob("history.json"), _MAX_CANDIDATES + 1))
    if len(candidates) > _MAX_CANDIDATES:
        raise ValueError(f"Too many histories: scan limited to {_MAX_CANDIDATES}")
    candidates.sort(key=lambda p: p.as_posix())
    if not candidates:
        raise ValueError(f"No history.json experiment artifacts under {root}")

    direction = mode or _metric_direction(metric)
    key = f"{split}_{metric}"
    rows, skipped = [], []
    for history_file in candidates:
        path = history_file.parent
        name = path.relative_to(root).as_posix()
        if history_file.is_symlink() or not history_file.resolve().is_relative_to(root):
            _error(skipped, name, "symlink or external history", strict=strict)
            continue
        manifest_file = path / "manifest.json"
        status = "history-only"
        if manifest_file.exists():
            if manifest_file.is_symlink():
                _error(skipped, name, "symlinked manifest", strict=strict)
                continue
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest is not a JSON object")
                status = manifest.get("status")
            except (OSError, ValueError, UnicodeError) as exc:
                _error(skipped, name, f"invalid manifest: {type(exc).__name__}", strict=strict)
                continue
            if status != "completed":
                _error(skipped, name, "manifest status is not completed", strict=strict)
                continue
        try:
            values = _values(_read_history(history_file), key)
            if not values:
                raise ValueError(f"no {key} values recorded")
        except (ValueError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            _error(skipped, name, f"invalid {key} history: {exc}", strict=strict)
            continue
        best = min(values) if direction == "min" else max(values)
        rows.append({
            "run": name, "status": status,
            "epochs": len(values), "best_epoch": values.index(best) + 1,
            "best": best, "final": values[-1],
        })
    if not rows:
        raise ValueError(f"No comparable {key} histories under {root}")

    sign = 1 if direction == "min" else -1
    rows.sort(key=lambda row: (sign * row[rank_by], row["run"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "metric": metric, "split": split, "mode": direction,
        "rank_by": rank_by, "total": len(rows), "total_candidates": len(candidates),
        "runs": rows[:top], "skipped": skipped,
    }


def _escape_cell(value):
    return (str(value).replace("\\", "\\\\").replace("|", "\\|")
            .replace("\r", " ").replace("\n", " ")
            .replace("<", "&lt;").replace(">", "&gt;")
            .replace(chr(96), "\\" + chr(96)))


def format_leaderboard(summary):
    """Build a readable Markdown leaderboard from aggregate epoch metrics."""
    lines = [
        f"### Experiment leaderboard — {summary['split']}_{summary['metric']}",
        "",
        f"**Optimize:** {summary['mode']} · **Rank by:** {summary['rank_by']} "
        f"· **Comparable runs:** {summary['total']}",
        "",
        "| Rank | Run | Status | Epochs | Best epoch | Best | Final |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        lines.append(
            f"| {row['rank']} | {_escape_cell(row['run'])} | "
            f"{row['status']} | {row['epochs']} | {row['best_epoch']} | "
            f"{row['best']:.6g} | {row['final']:.6g} |"
        )
    if summary["skipped"]:
        lines.extend([
            "",
            f"**Skipped {len(summary['skipped'])} incompatible or incomplete "
            "run(s)** (details available in JSON).",
        ])
    lines.extend([
        "",
        "_Ranks summarize recorded metrics only; compare like-for-like data "
        "splits before drawing scientific conclusions._",
    ])
    return "\n".join(lines)


def save_leaderboard(summary, output):
    """Save aggregate rankings to .md, .json, or .csv. Never copy datasets."""
    target = Path(output).expanduser()
    suffix = target.suffix.lower()
    if suffix not in (".md", ".json", ".csv"):
        raise ValueError("Leaderboard output must end in .md, .json, or .csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".md":
        target.write_text(format_leaderboard(summary) + "\n", encoding="utf-8")
    elif suffix == ".json":
        target.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    else:
        columns = ["rank", "run", "status", "epochs", "best_epoch", "best", "final"]
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary["runs"])
    return target


def show_leaderboard(root, *, metric="loss", split="val", mode=None,
                     rank_by="best", top=10, strict=False, save=None):
    """Show an experiment leaderboard in Jupyter or terminal Hy."""
    report = discover_leaderboard(
        root, metric=metric, split=split, mode=mode,
        rank_by=rank_by, top=top, strict=strict,
    )
    if save is not None:
        save_leaderboard(report, save)
    markdown = format_leaderboard(report)
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is not None and getattr(ipython, "kernel", None) is not None:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    else:
        print(markdown)
    return report
