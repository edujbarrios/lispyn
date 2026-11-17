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



def balanced_sample_weights(labels, *, max_classes=30):
    """Return one weight per input observation, for weighted random sampling.

    Each occurrence of a class gets n_samples / (n_classes * class_count).
    The ordering matches the input exactly. Values are returned in memory
    only; no labels, records or weights are written or displayed.
    These weights can be given to PyTorch WeightedRandomSampler as a
    one-dimensional sequence, independent of the class index encoding.
    """
    report, keys = _count_labels(
        labels, title="Training labels", max_classes=max_classes, _keys=True
    )
    total, n_classes = report["total"], report["num_classes"]
    lookup = {
        key: total / (n_classes * row["count"])
        for key, row in zip(keys, report["classes"])
    }
    result = []
    for raw_value in labels:
        value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        try:
            result.append(lookup[(type(value), value)])
        except KeyError as exc:
            raise ValueError("labels changed during sample-weight calculation") from exc
    if len(result) != total:
        raise ValueError("labels changed during sample-weight calculation")
    return result


def compare_label_splits(train_labels, eval_labels, *,
                         title="Classification label shift", max_classes=30,
                         save=None):
    """Compare train and evaluation label coverage and aggregate distributions.

    Total variation distance and Jensen-Shannon divergence (bits, range 0-1)
    use the union of classes, treating missing classes as zero observations.
    This is descriptive, not a calibrated hypothesis test. Only class-level
    summaries are returned, printed and optionally exported.
    """
    title = _validate_text("title", title)
    train, train_keys = _count_labels(
        train_labels, title="Train", max_classes=max_classes, _keys=True
    )
    evaluation, eval_keys = _count_labels(
        eval_labels, title="Evaluation", max_classes=max_classes, _keys=True
    )
    train_rows = dict(zip(train_keys, train["classes"]))
    eval_rows = dict(zip(eval_keys, evaluation["classes"]))
    keys = list(train_keys) + [key for key in eval_keys if key not in train_rows]
    if len(keys) > max_classes:
        raise ValueError(f"combined splits have more than {max_classes} distinct classes")

    shown = {}
    for key in keys:
        label = (train_rows.get(key) or eval_rows[key])["label"]
        if label in shown and shown[label] != key:
            raise ValueError("different class labels must not have identical display names")
        shown[label] = key

    classes = []
    total_variation = 0.0
    js_divergence = 0.0
    for key in keys:
        left = train_rows.get(key, {"count": 0})["count"]
        right = eval_rows.get(key, {"count": 0})["count"]
        p = left / train["total"]
        q = right / evaluation["total"]
        midpoint = (p + q) / 2
        total_variation += abs(p - q) / 2
        if p:
            js_divergence += p * math.log2(p / midpoint) / 2
        if q:
            js_divergence += q * math.log2(q / midpoint) / 2
        classes.append({
            "label": (train_rows.get(key) or eval_rows[key])["label"],
            "train_count": left, "eval_count": right,
            "train_share": p, "eval_share": q,
            "delta_share": q - p,
        })

    result = {
        "title": title,
        "train_total": train["total"], "eval_total": evaluation["total"],
        "num_classes": len(keys),
        "unseen_in_train": [row["label"] for row in classes if row["train_count"] == 0],
        "missing_in_eval": [row["label"] for row in classes if row["eval_count"] == 0],
        "total_variation_distance": min(1.0, max(0.0, total_variation)),
        "js_divergence": min(1.0, max(0.0, js_divergence)),
        "classes": classes,
    }
    markdown_rows = [
        f"### {_cell(title)}", "",
        "| Class | Train count | Eval count | Train share | Eval share | Change (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    markdown_rows.extend(
        f"| {_cell(row['label'])} | {row['train_count']} | {row['eval_count']} | "
        f"{row['train_share']:.1%} | {row['eval_share']:.1%} | "
        f"{row['delta_share'] * 100:+.1f} |"
        for row in classes
    )
    markdown_rows.extend([
        "",
        f"**Train samples:** {train['total']} · **Eval samples:** {evaluation['total']}",
        f"**Total variation distance:** {result['total_variation_distance']:.4f}"
        f" · **Jensen-Shannon divergence (bits):** {result['js_divergence']:.4f}",
        f"**Classes unseen in training:** {len(result['unseen_in_train'])}"
        f" · **Classes absent from evaluation:** {len(result['missing_in_eval'])}",
    ])
    markdown = "\n".join(markdown_rows)
    if save is not None:
        output = Path(save).expanduser()
        if output.suffix.lower() not in (".md", ".json"):
            raise ValueError("label split report must be .md or .json")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        else:
            output.write_text(markdown + "\n", encoding="utf-8")
    from .narrative import _present
    _present(markdown, None)
    return result

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
