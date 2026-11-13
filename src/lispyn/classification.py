"""Classification evaluation for PyTorch/Jupyter without a sklearn dependency.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Confusion matrices use existing labels and predictions; this module does
not run inference, train a model, or download anything.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from .plots import _design, _figure, _positive


def _scalar(value):
    """Convert scalar tensor/NumPy values to ordinary Python label values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def confusion_counts(actual, predicted, *, classes=None):
    """Return (counts, classes) with rows=actual and columns=predicted.

    The original order of class labels is preserved unless classes explicitly
    specifies the desired order. Mis-typed or unseen classes are rejected
    instead of silently dropping observations.
    """
    if isinstance(actual, (str, bytes)) or isinstance(predicted, (str, bytes)):
        raise ValueError("actual and predicted must be sequences of class labels")
    try:
        actual = [_scalar(x) for x in actual]
        predicted = [_scalar(x) for x in predicted]
    except (TypeError, ValueError) as exc:
        raise ValueError("actual and predicted must be one-dimensional sequences") from exc
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal, nonzero lengths")

    observed = []
    try:
        for label in actual + predicted:
            hash(label)
            if label not in observed:
                observed.append(label)
    except TypeError as exc:
        raise ValueError("class labels must be hashable scalar values") from exc

    if classes is None:
        classes = observed
    else:
        if isinstance(classes, (str, bytes)):
            raise ValueError("classes must be an ordered sequence, not a string")
        try:
            classes = [_scalar(label) for label in classes]
            unique = set(classes)
        except (TypeError, ValueError) as exc:
            raise ValueError("classes must contain hashable scalar labels") from exc
        if not classes or len(unique) != len(classes):
            raise ValueError("classes must contain unique, nonempty class labels")
        if set(observed) - set(classes):
            raise ValueError("classes must include every observed class")

    positions = {label: idx for idx, label in enumerate(classes)}
    counts = [[0 for _ in classes] for _ in classes]
    for truth, guess in zip(actual, predicted):
        counts[positions[truth]][positions[guess]] += 1
    return counts, classes


def normalized_counts(counts, mode=None):
    """Normalize counts by actual row, predicted column, or all samples."""
    if mode not in (None, "true", "pred", "all"):
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'")
    if mode is None:
        return [[int(value) for value in row] for row in counts]
    total = sum(sum(row) for row in counts)
    row_sums = [sum(row) for row in counts]
    col_sums = [sum(row[idx] for row in counts) for idx in range(len(counts))]
    result = []
    for row_idx, row in enumerate(counts):
        out = []
        for col_idx, value in enumerate(row):
            denom = (
                row_sums[row_idx] if mode == "true"
                else col_sums[col_idx] if mode == "pred"
                else total
            )
            out.append(value / denom if denom else 0.0)
        result.append(out)
    return result


def plot_confusion_matrix(actual, predicted, *, classes=None, normalize=None,
                          theme="docusaurus-dark", palette="turquoise-spectrum",
                          title="Confusion matrix", subtitle=None,
                          xlabel="Predicted", ylabel="Actual",
                          figsize=(7, 5.6), title_size=16, label_size=11,
                          tick_size=10, annotate=True, colorbar=True,
                          save=None, dpi=150):
    """Render a themed confusion matrix from actual/predicted classification labels.

    Returns Matplotlib Figure for customization or inline Jupyter display.
    'true' normalization divides by row support, 'pred' by column support,
    'all' by total samples. Empty denominators are displayed as zero.
    """
    counts, names = confusion_counts(actual, predicted, classes=classes)
    matrix = normalized_counts(counts, normalize)
    if not isinstance(annotate, bool) or not isinstance(colorbar, bool):
        raise ValueError("annotate and colorbar must be booleans")
    _positive("dpi", dpi, 600)
    design = _design(theme=theme, palette=palette)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise RuntimeError('Confusion matrix plotting requires: pip install -e ".[plots]"') from exc

    # All shades derive from the requested palette; the lowest intensities
    # blend into the dark axes while the highest become bright turquoise.
    from .plots import PALETTES
    swatches = PALETTES[palette] if palette is not None else (
        design["axes_bg"], design["train_color"], design["val_color"]
    )
    cmap = LinearSegmentedColormap.from_list(
        "lispyn-confusion", [design["axes_bg"], *swatches]
    )
    fig, ax = _figure(
        figsize=figsize, design=design, title=title, subtitle=subtitle,
        xlabel=xlabel, ylabel=ylabel, grid=False, title_size=title_size,
        label_size=label_size, tick_size=tick_size,
    )
    maximum = max(max(row) for row in matrix)
    maximum = max(maximum, 1 if normalize is not None else 1)
    image = ax.imshow(matrix, interpolation="nearest", cmap=cmap,
                      vmin=0, vmax=maximum, aspect="equal")
    ax.set_xticks(list(range(len(names))))
    ax.set_yticks(list(range(len(names))))
    ax.set_xticklabels([str(x) for x in names], rotation=30, ha="right")
    ax.set_yticklabels([str(x) for x in names])
    ax.tick_params(length=0, colors=design["text_color"])
    if annotate:
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                text_color = (
                    "#152129" if value / maximum > 0.58
                    else design["text_color"]
                )
                text = f"{value:.2f}" if normalize is not None else str(value)
                ax.text(col_idx, row_idx, text, color=text_color,
                        ha="center", va="center", fontsize=tick_size)
    if colorbar:
        colorbar_obj = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.035)
        colorbar_obj.ax.tick_params(colors=design["text_color"])
        colorbar_obj.outline.set_edgecolor(design["spine_color"])
    fig.tight_layout(rect=(0, 0, 1, 0.93 if fig._suptitle else 1))
    if save is not None:
        output = Path(save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, facecolor=fig.get_facecolor(), dpi=dpi)
    return fig


def make_confusion_plotter(**defaults):
    """Return a reusable confusion-matrix plotter, customizable per call."""
    return partial(plot_confusion_matrix, **defaults)
