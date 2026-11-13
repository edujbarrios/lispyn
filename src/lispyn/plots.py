"""Simple training curves for Jupyter and command-line reports.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Matplotlib is optional and imported only when plotting is requested.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path


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


def plot_history(source, *, metric="loss", save=None):
    """Plot train/validation curves from a Run, dict or history.json file.

    Example (Hy):
        (plot-history run :metric "accuracy" :save "plots/accuracy.png")

    Returns a Matplotlib Figure for display or customization. Calling this
    function does not alter the trained model or perform any model evaluation.
    """
    if not isinstance(metric, str) or not metric or metric.startswith(("train_", "val_")):
        raise ValueError("metric must be an unprefixed name such as 'loss' or 'accuracy'")
    history = _read_history(source)
    train = _values(history, f"train_{metric}")
    valid = _values(history, f"val_{metric}")
    if not train and not valid:
        raise ValueError(f"No recorded values for metric: {metric}")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires: pip install -e ".[plots]"') from exc

    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot(range(1, len(train) + 1), train, marker=".", label="Train")
    if valid:
        ax.plot(range(1, len(valid) + 1), valid, marker=".", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(f"Lispyn · {metric.replace('_', ' ').capitalize()}")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    if save is not None:
        output = Path(save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    return fig
