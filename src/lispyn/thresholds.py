"""Aggregate binary probability-threshold diagnostics for Hy/Jupyter.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
This module never trains a model or exports individual labels and scores.
Choose thresholds using held-out validation data, never held-out test labels.
"""
from __future__ import annotations

from bisect import bisect_left
import html
import json
import math
from pathlib import Path


def _scalar(value, name):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(f"{name} must contain scalar values") from exc
    return value


def _vector(values, name):
    if isinstance(values, (str, bytes, dict)) or not hasattr(values, "__len__"):
        raise TypeError(f"{name} must be a nonempty one-dimensional sequence")
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 0 < len(values) <= 1_000_000:
        raise ValueError(f"{name} must contain between 1 and 1,000,000 values")
    return [_scalar(value, name) for value in values]


def _threshold_grid(thresholds):
    if thresholds is None:
        return [step / 20 for step in range(1, 20)]
    values = _vector(thresholds, "thresholds")
    if len(values) > 101:
        raise ValueError("thresholds must contain at most 101 values")
    result = []
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("thresholds must be finite numbers between 0 and 1")
        result.append(float(value))
    if len(set(result)) != len(result):
        raise ValueError("thresholds must be unique")
    return sorted(result)


def sweep_binary_thresholds(actual, scores, *, thresholds=None,
                            optimize="f1", title="Binary threshold evaluation"):
    """Summarize validation metrics for each proposed positive-score cutoff.

    Predictions are positive when score >= threshold. Both binary classes must
    be represented; zero-denominator metrics are 0.0. Ties are resolved toward
    the cutoff nearest 0.5, then the smaller cutoff. The default grid contains
    0.05, 0.10, ..., 0.95 and is not a globally exhaustive search.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be nonempty text")
    if optimize not in ("f1", "balanced_accuracy"):
        raise ValueError("optimize must be 'f1' or 'balanced_accuracy'")
    labels = _vector(actual, "actual")
    probabilities = _vector(scores, "scores")
    if len(labels) != len(probabilities):
        raise ValueError("actual and scores must have equal lengths")
    for label in labels:
        if not isinstance(label, (bool, int)) or label not in (0, 1):
            raise ValueError("actual must contain only binary integer labels 0 and 1")
    for value in probabilities:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("scores must be finite probabilities between 0 and 1")
    positives = sorted(float(score) for label, score in zip(labels, probabilities) if label == 1)
    negatives = sorted(float(score) for label, score in zip(labels, probabilities) if label == 0)
    if not positives or not negatives:
        raise ValueError("actual must contain both positive and negative labels")

    total = len(labels)
    rows = []
    for threshold in _threshold_grid(thresholds):
        tp = len(positives) - bisect_left(positives, threshold)
        fp = len(negatives) - bisect_left(negatives, threshold)
        fn, tn = len(positives) - tp, len(negatives) - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(positives)
        specificity = tn / len(negatives)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        rows.append({
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": f1, "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": (tp + tn) / total,
        })
    selected = max(rows, key=lambda row: (
        row[optimize], -abs(row["threshold"] - 0.5), -row["threshold"]
    ))
    return {
        "title": title.strip(), "samples": total,
        "positives": len(positives), "negatives": len(negatives),
        "optimize": optimize, "best_threshold": selected["threshold"],
        "best_metrics": dict(selected), "rows": rows,
    }


def format_threshold_sweep(report):
    """Produce a small Markdown table of aggregate threshold metrics."""
    title = html.escape(report["title"]).replace("|", r"\|").replace("\n", " ")
    rows = [
        f"### {title}", "",
        f"**Samples:** {report['samples']} · **Positives:** {report['positives']}"
        f" · **Negatives:** {report['negatives']}",
        f"**Selected threshold:** {report['best_threshold']:.2f}"
        f" (best {report['optimize']} on the specified grid)", "",
        "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 | Balanced accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        rows.append(
            f"| {row['threshold']:.2f} | {row['tp']} | {row['fp']} | {row['fn']}"
            f" | {row['tn']} | {row['precision']:.3f} | {row['recall']:.3f}"
            f" | {row['f1']:.3f} | {row['balanced_accuracy']:.3f} |"
        )
    return "\n".join(rows)


def threshold_report(actual, scores, *, thresholds=None, optimize="f1",
                     title="Binary threshold evaluation", save=None):
    """Show an aggregate report in terminal/Jupyter; optionally save MD/JSON."""
    report = sweep_binary_thresholds(
        actual, scores, thresholds=thresholds, optimize=optimize, title=title
    )
    markdown = format_threshold_sweep(report)
    if save is not None:
        output = Path(save).expanduser()
        if output.suffix.lower() not in (".md", ".json"):
            raise ValueError("threshold report must be saved as .md or .json")
        output.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
                if output.suffix.lower() == ".json" else markdown + "\n")
        output.write_text(data, encoding="utf-8")
    from .narrative import _present
    _present(markdown, None)
    return report


def plot_threshold_metrics(actual, scores, *, thresholds=None, optimize="f1",
                           title="Binary threshold evaluation", theme="paper",
                           figsize=(8, 4.5), dpi=150, save=None):
    """Plot precision, recall, F1 and balanced accuracy over threshold cutoffs."""
    report = sweep_binary_thresholds(
        actual, scores, thresholds=thresholds, optimize=optimize, title=title
    )
    from .plots import _design, _positive
    _positive("dpi", dpi, 600)
    if (not isinstance(figsize, (tuple, list)) or len(figsize) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 or v > 24 for v in figsize)):
        raise ValueError("figsize must contain two finite positive sizes at most 24")
    output = None
    if save is not None:
        output = Path(save).expanduser()
        if output.suffix.lower() not in (".png", ".svg"):
            raise ValueError("threshold plot must be saved as .png or .svg")
    design = _design(theme=theme)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(design["figure_bg"])
    ax.set_facecolor(design["axes_bg"])
    cutoffs = [row["threshold"] for row in report["rows"]]
    for key, label, style in (
        ("precision", "Precision", "-"),
        ("recall", "Recall", "--"),
        ("f1", "F1", "-."),
        ("balanced_accuracy", "Balanced accuracy", ":"),
    ):
        ax.plot(cutoffs, [row[key] for row in report["rows"]],
                label=label, linestyle=style, linewidth=2)
    ax.axvline(report["best_threshold"], color=design["text_color"],
               linestyle=":", alpha=0.65,
               label=f"Selected: {report['best_threshold']:.2f}")
    ax.set(title=report["title"], xlabel="Positive probability threshold",
           ylabel="Metric")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=design["text_color"])
    ax.title.set_color(design["text_color"])
    ax.xaxis.label.set_color(design["text_color"])
    ax.yaxis.label.set_color(design["text_color"])
    for spine in ax.spines.values():
        spine.set_color(design["spine_color"])
    ax.yaxis.grid(True, color=design["grid_color"], alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    return fig
