"""Class distribution summaries and figures for AI dataset notebooks.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
No model training, downloads or raw sample export are performed.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _cell(value):
    """Treat class labels as data, not HTML or Markdown markup."""
    return html.escape(str(value)).replace("|", r"\|").replace("\n", " ").replace("\r", " ")


def _validate_text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _count_labels(labels, *, title, max_classes, _keys=False):
    title = _validate_text("title", title)
    if isinstance(max_classes, bool) or not isinstance(max_classes, int) or not 1 <= max_classes <= 50:
        raise ValueError("max_classes must be an integer from 1 to 50")
    if isinstance(labels, (str, bytes, dict)) or not hasattr(labels, "__len__"):
        raise TypeError("labels must be a finite, one-dimensional sequence")
    if getattr(labels, "ndim", 1) != 1:
        raise ValueError("labels must be a one-dimensional sequence")
    if len(labels) == 0:
        raise ValueError("labels cannot be empty")

    counts = {}
    rendered = {}
    for value in labels:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError("class labels must be scalar values") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("labels must not contain NaN or infinity")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("labels must be scalar strings, integers, floats or booleans")
        shown = str(value)
        if not shown.strip() or len(shown) > 120:
            raise ValueError("class labels must be nonempty and at most 120 characters")
        key = (type(value), value)
        if key not in counts:
            if shown in rendered and rendered[shown] != key:
                raise ValueError("different class labels must not have identical display names")
            if len(counts) >= max_classes:
                raise ValueError(f"more than {max_classes} distinct classes; increase max_classes")
            rendered[shown] = key
            counts[key] = {"label": shown, "count": 0}
        counts[key]["count"] += 1
    total = sum(row["count"] for row in counts.values())
    classes = [
        {"label": row["label"], "count": row["count"], "share": row["count"] / total}
        for row in counts.values()
    ]
    values = [row["count"] for row in classes]
    result = {
        "title": title, "total": total, "num_classes": len(classes),
        "majority_share": max(values) / total,
        "imbalance_ratio": max(values) / min(values),
        "classes": classes,
    }
    return (result, tuple(counts)) if _keys else result


def label_balance(labels, *, title="Label distribution", max_classes=30, save=None):
    """Display and return aggregate class counts and shares (first-seen order).

    Saves Markdown or JSON if requested. No observation-level data is stored.
    Class names themselves remain visible, so review exports before sharing.
    """
    report = _count_labels(labels, title=title, max_classes=max_classes)
    rows = [
        f"### {_cell(report['title'])}", "",
        "| Class | Count | Share |", "| --- | ---: | ---: |",
    ]
    rows.extend(
        f"| {_cell(row['label'])} | {row['count']} | {row['share']:.1%} |"
        for row in report["classes"]
    )
    rows.extend([
        "",
        f"**Total:** {report['total']} · **Classes:** {report['num_classes']}"
        f" · **Majority share:** {report['majority_share']:.1%}"
        f" · **Imbalance ratio:** {report['imbalance_ratio']:.2f}:1",
    ])
    markdown = "\n".join(rows)
    if save is not None:
        output = Path(save).expanduser()
        if output.suffix.lower() not in (".json", ".md"):
            raise ValueError("class balance report must be .json or .md")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        else:
            output.write_text(markdown + "\n", encoding="utf-8")
    from .narrative import _present
    _present(markdown, None)
    return report



def balanced_class_weights(labels, *, class_order=None, max_classes=30):
    """Return inverse-frequency class weights aligned with a documented order.

    The balanced formula is n_samples / (n_classes * class_count).
    Pass class_order in the exact order of the model's encoded target indices
    (0 through n_classes - 1). Without it, classes retain first-seen order.
    No PyTorch dependency, raw samples, or model modifications are required.
    """
    report, observed_keys = _count_labels(
        labels, title="Training labels", max_classes=max_classes, _keys=True
    )
    rows = dict(zip(observed_keys, report["classes"]))
    if class_order is None:
        ordered_keys = observed_keys
    else:
        supplied, ordered_keys = _count_labels(
            class_order, title="Class order", max_classes=max_classes, _keys=True
        )
        if (supplied["num_classes"] != report["num_classes"]
                or any(row["count"] != 1 for row in supplied["classes"])
                or set(ordered_keys) != set(observed_keys)):
            raise ValueError("class_order must contain each observed class exactly once")

    chosen = [rows[key] for key in ordered_keys]
    total = report["total"]
    num_classes = report["num_classes"]
    return {
        "class_order": [row["label"] for row in chosen],
        "counts": [row["count"] for row in chosen],
        "weights": [total / (num_classes * row["count"]) for row in chosen],
        "total": total,
        "num_classes": num_classes,
    }


def plot_label_balance(labels, *, title="Label distribution",
                       max_classes=30, theme="paper", figsize=(8, 4.5),
                       dpi=150, save=None):
    """Return a Matplotlib Figure with one bar per observed class."""
    report = _count_labels(labels, title=title, max_classes=max_classes)
    from .plots import _design, _positive

    _positive("dpi", dpi, 600)
    if (not isinstance(figsize, (tuple, list)) or len(figsize) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 or v > 24 for v in figsize)):
        raise ValueError("figsize must contain two positive finite sizes at most 24")
    output = None
    if save is not None:
        output = Path(save).expanduser()
        if output.suffix.lower() not in (".png", ".svg"):
            raise ValueError("class balance plot must be .png or .svg")
    design = _design(theme=theme)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(design["figure_bg"])
    ax.set_facecolor(design["axes_bg"])
    names = [row["label"] for row in report["classes"]]
    amounts = [row["count"] for row in report["classes"]]
    bars = ax.bar(range(len(names)), amounts, color=design["train_color"])
    ax.set_title(report["title"], color=design["text_color"])
    ax.set_ylabel("Samples", color=design["text_color"])
    ax.set_xlabel("Class", color=design["text_color"])
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.tick_params(colors=design["text_color"])
    for side in ax.spines.values():
        side.set_color(design["spine_color"])
    ax.yaxis.grid(True, color=design["grid_color"], alpha=0.35)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, amounts):
        ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, count),
                    ha="center", va="bottom", color=design["text_color"])
    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    return fig
