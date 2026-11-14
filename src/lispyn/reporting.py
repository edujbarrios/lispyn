"""Privacy-conscious, shareable Markdown reports for Lispyn Hy experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Only aggregate metric histories are summarized. Model weights, data samples,
predictions, dataset paths and full manifests are not copied into reports.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path


def _metric_direction(metric: str) -> str:
    """Select the typical optimization direction for a metric name."""
    name = metric.lower()
    if (name.endswith("loss") or name.endswith("error")
            or name.endswith(("mae", "mse", "rmse", "perplexity", "wer", "cer"))):
        return "min"
    return "max"


def _metric_history(history, key):
    values = history[key]
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Metric {key!r} needs numeric epoch values")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"Metric {key!r} needs numeric epoch values") from exc
    if not values:
        raise ValueError(f"Metric {key!r} cannot be empty")
    if any(isinstance(x, bool) or not isinstance(x, Real)
           or not math.isfinite(x) for x in values):
        raise ValueError(f"Metric {key!r} must contain only finite real values")
    return [float(value) for value in values]


def _safe_table(value):
    """Prevent Markdown table cells from breaking layout."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_run_report(run, *, title="Experiment results", notes=None,
                      metric=None, mode=None):
    """Return publication-friendly Markdown for a Lispyn Run or history.

    Summarizes best and final metrics, optimal epoch and direction. The source
    can be a workflow.Run (with .history) or a history mapping. Use metric=...
    to select a single named history and mode='min'/'max' to override the
    inferred optimization direction. No model weights or dataset examples
    are exposed.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be nonempty text")
    if notes is not None and not isinstance(notes, str):
        raise TypeError("notes must be a string")
    if mode not in (None, "min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    history = getattr(run, "history", run)
    if not isinstance(history, Mapping):
        raise TypeError("run must contain a mapping of metric histories")
    if metric is not None:
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric must be a nonempty string")
        if metric not in history:
            raise ValueError(f"Metric {metric!r} is absent from this run")
        keys = [metric]
    else:
        # Training may not have a validation dataset; workflow.fit leaves
        # unused history series (e.g. val_loss) as empty lists. Omit those
        # from a multi-metric narrative instead of rejecting a valid run.
        # Explicit metric=... still validates the chosen series strictly.
        keys = [
            key for key in history
            if isinstance(key, str)
            and key.startswith(("train_", "val_", "test_"))
            and not (hasattr(history[key], "__len__")
                     and len(history[key]) == 0)
        ]
    if not keys:
        raise ValueError("No training/validation metric histories found")

    rows = []
    max_epochs = 0
    for key in keys:
        values = _metric_history(history, key)
        max_epochs = max(max_epochs, len(values))
        direction = mode or _metric_direction(key)
        best = min(values) if direction == "min" else max(values)
        epoch = values.index(best) + 1
        rows.append(
            f"| {_safe_table(key)} | {direction} | {epoch} | "
            f"{best:.6g} | {values[-1]:.6g} |"
        )

    result = [
        f"# {title.strip()}",
        "",
        f"**Epochs recorded:** {max_epochs}",
    ]
    if hasattr(run, "stopped_early"):
        result.append(
            "**Early stopping:** " + (
                "yes" if run.stopped_early else "no"
            )
        )
    if notes:
        result.extend(["", "## Notes", "", notes.strip()])
    result.extend([
        "",
        "## Metrics",
        "",
        "| Metric | Optimize | Best epoch | Best value | Final value |",
        "| --- | --- | ---: | ---: | ---: |",
        *rows,
        "",
        "_Best epochs are 1-based. This report contains only metric summaries, "
        "not raw dataset rows, predictions, or model weights._",
    ])
    return "\n".join(result) + "\n"


def show_run_report(run, *, title="Experiment results", notes=None,
                    metric=None, mode=None, save=None):
    """Display a Markdown report in Jupyter; print it when running Hy in CLI.

    Optional save='report.md' exports a standalone Markdown artifact.
    Returns the Markdown string for further customization and reuse.
    """
    report = render_run_report(
        run, title=title, notes=notes, metric=metric, mode=mode
    )
    if save is not None:
        path = Path(save).expanduser()
        if path.suffix.lower() != ".md":
            raise ValueError("Experiment report must have a .md extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is not None and getattr(ipython, "kernel", None) is not None:
        from IPython.display import Markdown, display
        display(Markdown(report))
    else:
        print(report)
    return report
