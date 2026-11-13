"""Reusable Matplotlib styles and multi-run training curves for Lispyn/Hy.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Matplotlib is an optional runtime dependency; no global styles are modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
import math
from pathlib import Path


# The original theme keys remain part of the public Python plotting API.
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
    "docusaurus-dark": {
        "background": "#1B1B1D", "axes": "#23272F",
        "foreground": "#E6EDF3", "grid": "#3A3F4B",
        "spine": "#4B5563", "legend_bg": "#23272F",
        "legend_edge": "#4B5563",
        "train": "#40E0D0", "validation": "#8CF0EB",
    },
}

PALETTES = {
    "turquoise-spectrum": (
        "#004D4D", "#006666", "#008080", "#40E0D0",
        "#66E6DD", "#8CF0EB", "#B2FAF9",
    ),
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


def _positive(name, value, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be positive and finite" +
                         (f" (at most {maximum})" if maximum else ""))
    return value


def _unit_interval(name, value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _design(*, theme="clean", palette=None, figure_bg=None, axes_bg=None,
            grid_color=None, text_color=None, spine_color=None,
            legend_bg=None, legend_edge=None, train_color=None, val_color=None):
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}; choose: {', '.join(THEMES)}")
    if palette is not None and palette not in PALETTES:
        raise ValueError(f"Unknown palette {palette!r}; choose: {', '.join(PALETTES)}")
    try:
        from matplotlib.colors import is_color_like
    except ImportError as exc:
        raise RuntimeError('Plotting requires: pip install -e ".[plots]"') from exc

    base = THEMES[theme]
    chosen = PALETTES[palette] if palette is not None else None
    design = {
        "figure_bg": figure_bg or base["background"],
        "axes_bg": axes_bg or base["axes"],
        "grid_color": grid_color or base["grid"],
        "text_color": text_color or base["foreground"],
        "spine_color": spine_color or base.get("spine", base["grid"]),
        "legend_bg": legend_bg or base.get("legend_bg", base["axes"]),
        "legend_edge": legend_edge or base.get("legend_edge", base["grid"]),
        "train_color": train_color or (chosen[3] if chosen else base["train"]),
        "val_color": val_color or (chosen[5] if chosen else base["validation"]),
    }
    for key, color in design.items():
        if not is_color_like(color):
            raise ValueError(f"{key} is not a valid Matplotlib color: {color!r}")
    return design


def _figure(*, figsize, design, xlabel, ylabel, title, subtitle,
            title_size, label_size, tick_size, grid):
    import matplotlib.pyplot as plt

    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a pair of positive finite dimensions <=40")
    for dimension in figsize:
        _positive("figsize", dimension, 40)
    for name, value in (("title_size", title_size), ("label_size", label_size),
                        ("tick_size", tick_size)):
        _positive(name, value, 80)
    fig, ax = plt.subplots(figsize=figsize, facecolor=design["figure_bg"])
    ax.set_facecolor(design["axes_bg"])
    for spine in ax.spines.values():
        spine.set_color(design["spine_color"])
    ax.tick_params(axis="both", colors=design["text_color"], labelsize=tick_size)
    ax.set_xlabel(xlabel, color=design["text_color"], fontsize=label_size)
    ax.set_ylabel(ylabel, color=design["text_color"], fontsize=label_size)
    if subtitle:
        # Main title belongs to the figure; subtitle stays directly above the axes.
        fig.suptitle(title, color=design["text_color"], fontsize=title_size,
                     fontweight="semibold", y=0.98)
        ax.set_title(subtitle, color=design["text_color"],
                     fontsize=max(9, label_size - 1), pad=13)
    else:
        ax.set_title(title, color=design["text_color"],
                     fontsize=title_size, fontweight="semibold", pad=16)
    if grid:
        ax.grid(True, color=design["grid_color"], alpha=0.55, linewidth=0.8)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return fig, ax


def _finalize(fig, ax, *, design, legend, dpi, save):
    _positive("dpi", dpi, 600)
    if legend and ax.lines:
        ax.legend(facecolor=design["legend_bg"], edgecolor=design["legend_edge"],
                  labelcolor=design["text_color"], framealpha=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if fig._suptitle else 1))
    if save is not None:
        output = Path(save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, facecolor=fig.get_facecolor(), dpi=dpi)
    return fig


def make_plotter(**defaults):
    """Reusable plot function; per-call options override defaults.

    This is the Python function emitted by Hy's defcurve macro.
    """
    return partial(plot_history, **defaults)


def plot_history(source, *, metric="loss", save=None, theme="clean",
                 title=None, subtitle=None, xlabel="Epoch", ylabel=None,
                 train_label="Training", val_label="Validation",
                 train_color=None, val_color=None, palette=None,
                 figure_bg=None, axes_bg=None, grid_color=None,
                 text_color=None, spine_color=None, legend_bg=None,
                 legend_edge=None, figsize=(7, 4), linewidth=2.2,
                 line_width=None, marker=".", alpha=1.0, fill_alpha=0.0,
                 title_size=14, label_size=11, tick_size=10,
                 grid=True, legend=True, dpi=140):
    """Plot train and validation metrics with fully local Matplotlib styling.

    Supports the original options plus theme/palette, backgrounds, text,
    subtitle, markers and an optional shaded difference between curves.
    Returns a Matplotlib Figure; no model is trained or evaluated here.
    """
    if not isinstance(metric, str) or not metric or metric.startswith(("train_", "val_")):
        raise ValueError("metric must be an unprefixed name such as 'loss' or 'accuracy'")
    if line_width is not None:
        linewidth = line_width
    _positive("linewidth", linewidth, 40)
    _unit_interval("alpha", alpha)
    _unit_interval("fill_alpha", fill_alpha)
    history = _read_history(source)
    train = _values(history, f"train_{metric}")
    valid = _values(history, f"val_{metric}")
    if not train and not valid:
        raise ValueError(f"No recorded values for metric: {metric}")

    # Optional dependency is deliberately deferred until a chart is requested.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires: pip install -e ".[plots]"') from exc

    design = _design(
        theme=theme, palette=palette, figure_bg=figure_bg, axes_bg=axes_bg,
        grid_color=grid_color, text_color=text_color, spine_color=spine_color,
        legend_bg=legend_bg, legend_edge=legend_edge,
        train_color=train_color, val_color=val_color,
    )
    display_name = metric.replace("_", " ").capitalize()
    fig, ax = _figure(
        figsize=figsize, design=design, xlabel=xlabel,
        ylabel=ylabel if ylabel is not None else display_name,
        title=title if title is not None else f"Lispyn · {display_name}",
        subtitle=subtitle, title_size=title_size,
        label_size=label_size, tick_size=tick_size, grid=grid,
    )
    if train:
        ax.plot(range(1, len(train) + 1), train, marker=marker,
                linewidth=linewidth, alpha=alpha,
                color=design["train_color"], label=train_label)
    if valid:
        ax.plot(range(1, len(valid) + 1), valid, marker=marker,
                linewidth=linewidth, alpha=alpha,
                color=design["val_color"], label=val_label)
    if fill_alpha and train and valid:
        size = min(len(train), len(valid))
        ax.fill_between(range(1, size + 1), train[:size], valid[:size],
                        color=design["train_color"], alpha=fill_alpha)
    return _finalize(fig, ax, design=design, legend=legend, dpi=dpi, save=save)


def plot_comparison(runs, *, metric="loss", split="val", save=None,
                    theme="docusaurus-dark", palette="turquoise-spectrum",
                    title=None, subtitle=None, xlabel="Epoch", ylabel=None,
                    figsize=(9, 4.8), linewidth=2.5, marker=None, alpha=0.95,
                    title_size=15, label_size=11, tick_size=10,
                    grid=True, legend=True, dpi=140):
    """Plot the same metric from several named runs on one set of axes.

    runs: mapping of labels to Run objects, histories, or history.json paths.
    Only the requested split is displayed; different run lengths are allowed.
    """
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("runs must be a non-empty mapping of names to histories")
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if not isinstance(metric, str) or not metric or metric.startswith(("train_", "val_")):
        raise ValueError("metric must be an unprefixed name such as 'loss' or 'accuracy'")
    _positive("linewidth", linewidth, 40)
    _unit_interval("alpha", alpha)
    design = _design(theme=theme, palette=palette)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires: pip install -e ".[plots]"') from exc

    values = []
    for name, source in runs.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("run names must be non-empty strings")
        series = _values(_read_history(source), f"{split}_{metric}")
        if not series:
            raise ValueError(f"Missing {split}_{metric} for run {name!r}")
        values.append((name, series))

    display_name = metric.replace("_", " ").capitalize()
    fig, ax = _figure(
        figsize=figsize, design=design, xlabel=xlabel,
        ylabel=ylabel if ylabel is not None else display_name,
        title=title if title is not None else f"Compare {split} {display_name}",
        subtitle=subtitle, title_size=title_size, label_size=label_size,
        tick_size=tick_size, grid=grid,
    )
    colors = PALETTES[palette] if palette is not None else (
        design["train_color"], design["val_color"])
    # On a dark background, begin with the brighter turquoise shades.
    order = (3, 4, 5, 6, 2, 1, 0) if palette == "turquoise-spectrum" else tuple(
        range(len(colors)))
    for index, (name, series) in enumerate(values):
        ax.plot(range(1, len(series) + 1), series,
                color=colors[order[index % len(order)]], label=name,
                linewidth=linewidth, marker=marker, alpha=alpha)
    return _finalize(fig, ax, design=design, legend=legend, dpi=dpi, save=save)
