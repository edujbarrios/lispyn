"""Reusable, themed training curves for Jupyter and static exports.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Matplotlib is optional: it is imported only when a plot is requested.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
import math
from pathlib import Path


THEMES = {
    "clean": {
        "background": "#FFFFFF", "axes": "#FFFFFF",
        "foreground": "#17212F", "grid": "#CAD4DF",
        "train": "#2563EB", "validation": "#F97316",
    },
    "dark": {
        "background": "#141824", "axes": "#1B2230",
        "foreground": "#F0F4FC", "grid": "#465166",
        "train": "#38BDF8", "validation": "#FB923C",
    },
    "paper": {
        "background": "#FCFBF8", "axes": "#FCFBF8",
        "foreground": "#34312E", "grid": "#D3CDC2",
        "train": "#326B7D", "validation": "#B3603C",
    },
}


def _read_history(source):
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    elif hasattr(source, "history"):
        source = source.history
    if not isinstance(source, Mapping):
        raise TypeError("Expected a Run object, history mapping, or history.json path")
    return source


def _values(history, key):
    values = history.get(key, [])
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a sequence of numeric epoch values")
    validated = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} contains a non-numeric value")
        if not math.isfinite(value):
            raise ValueError(f"{key} contains non-finite values")
        validated.append(float(value))
    return validated


def make_plotter(**defaults):
    """Return a reusable plot function; call-specific options override defaults.

    Intended for Hy's defcurve macro. Also works from ordinary Python:
        dark_curve = make_plotter(theme="dark", train_color="#38BDF8")
        dark_curve(history, metric="accuracy", title="Validation accuracy")
    """
    return partial(plot_history, **defaults)


def plot_history(source, *, metric="loss", save=None, theme="clean",
                 title=None, xlabel="Epoch", ylabel=None,
                 train_label="Training", val_label="Validation",
                 train_color=None, val_color=None, figsize=(7, 4),
                 linewidth=2.2, grid=True, legend=True, dpi=140):
    """Plot existing train/validation histories using a customizable design.

    Accepts a Run object, history mapping, or history.json file. No training,
    evaluation, data decoding or global Matplotlib style mutation occurs.
    Returns a Figure for inline Jupyter display or further customizations.
    """
    if not isinstance(metric, str) or not metric or metric.startswith(("train_", "val_")):
        raise ValueError("metric must be an unprefixed name such as 'loss' or 'accuracy'")
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}; choose: {', '.join(THEMES)}")
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or
        not math.isfinite(x) or not 0 < x <= 40 for x in figsize
    ):
        raise ValueError("figsize must be a pair of positive finite dimensions <=40")
    if isinstance(linewidth, bool) or not isinstance(linewidth, (int, float)) or (
        not math.isfinite(linewidth) or linewidth <= 0
    ):
        raise ValueError("linewidth must be positive and finite")
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or (
        not math.isfinite(dpi) or not 0 < dpi <= 600
    ):
        raise ValueError("dpi must be positive and <=600")

    history = _read_history(source)
    train = _values(history, f"train_{metric}")
    valid = _values(history, f"val_{metric}")
    if not train and not valid:
        raise ValueError(f"No recorded values for metric: {metric}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import is_color_like
    except ImportError as exc:
        raise RuntimeError('Plotting requires: pip install -e ".[plots]"') from exc

    theme_colors = THEMES[theme]
    for name, color in (("train_color", train_color), ("val_color", val_color)):
        if color is not None and not is_color_like(color):
            raise ValueError(f"{name} is not a valid Matplotlib color: {color!r}")

    fig, ax = plt.subplots(figsize=figsize, facecolor=theme_colors["background"])
    ax.set_facecolor(theme_colors["axes"])
    foreground = theme_colors["foreground"]
    for spine in ax.spines.values():
        spine.set_color(theme_colors["grid"])
    ax.tick_params(axis="both", colors=foreground)
    if train:
        ax.plot(range(1, len(train) + 1), train, marker=".", linewidth=linewidth,
                color=train_color or theme_colors["train"], label=train_label)
    if valid:
        ax.plot(range(1, len(valid) + 1), valid, marker=".", linewidth=linewidth,
                color=val_color or theme_colors["validation"], label=val_label)

    label = metric.replace("_", " ").capitalize()
    ax.set_xlabel(xlabel, color=foreground)
    ax.set_ylabel(ylabel if ylabel is not None else label, color=foreground)
    ax.set_title(title if title is not None else f"Lispyn · {label}",
                 color=foreground)
    ax.grid(bool(grid), color=theme_colors["grid"], alpha=0.45)
    if legend:
        ax.legend(facecolor=theme_colors["axes"], edgecolor=theme_colors["grid"],
                  labelcolor=foreground)
    fig.tight_layout()
    if save is not None:
        output = Path(save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, facecolor=fig.get_facecolor(), dpi=dpi)
    return fig
