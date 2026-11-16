"""Hy-first comparison reports and figures from saved native Lispyn runs.

The public macros call these functions with ordinary Python paths. Existing
history.json scores are read offline; no models or datasets are loaded, no
training occurs and no run source code is executed.
"""

from __future__ import annotations

from pathlib import Path
import json

from .comparison import compare_experiments, format_comparison, save_comparison
from .plots import plot_comparison


def _run_paths(directories):
    """Reject accidental strings and duplicate/ambiguous run directory names."""
    if isinstance(directories, (str, bytes, Path)):
        raise TypeError("runs must be a list/tuple of at least two run directories")
    if not isinstance(directories, (list, tuple)):
        raise TypeError("runs must be a list/tuple of run directories")
    if len(directories) < 2 or len(directories) > 20:
        raise ValueError("Compare between 2 and 20 run directories")
    paths = []
    for directory in directories:
        if not isinstance(directory, (str, Path)):
            raise TypeError("each run must be a path string or Path")
        path = Path(directory).expanduser()
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Run directory does not exist or is symlinked: {path}")
        if (path / "history.json").is_symlink():
            raise ValueError(f"Refusing a symlinked history.json: {path}")
        manifest_file = path / "manifest.json"
        if manifest_file.is_symlink():
            raise ValueError(f"Refusing a symlinked manifest.json: {path}")
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError, OSError) as exc:
                raise ValueError(f"Invalid manifest.json in {path}: {exc}") from exc
            if not isinstance(manifest, dict) or manifest.get("status") != "completed":
                raise ValueError(f"Run is not completed: {path}")
        paths.append(path)
    names = [path.name for path in paths]
    if any(not name or name in (".", "..") for name in names) or len(set(names)) != len(names):
        raise ValueError("Run directories must have distinct, nonempty names")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Run paths must identify different directories")
    return paths


def _display_markdown(markdown):
    """Display report in Jupyter or print when running native Hy."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None and getattr(shell, "kernel", None) is not None:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    else:
        print(markdown)


def show_run_comparison(runs, *, metric="loss", split="val", mode=None,
                        baseline=None, save=None):
    """Compare recorded scalar metrics and display Markdown; return raw summary.

    Positive best/final gain is always an improvement over the baseline
    according to the selected direction. The comparison is descriptive:
    different data splits or seeds are not statistically comparable.
    """
    paths = _run_paths(runs)
    result = compare_experiments(
        paths, metric=metric, split=split, mode=mode, baseline=baseline,
    )
    if save is not None:
        save_comparison(result, save)
    _display_markdown(format_comparison(result))
    return result


def plot_run_comparison(runs, *, metric="loss", split="val",
                        save=None, **plot_options):
    """Visualize named recorded histories; returns a Matplotlib Figure.

    Only the same split and metric are drawn from each saved history.
    Uses the existing Lispyn styling API (theme/palette/figsize/title/...).
    """
    paths = _run_paths(runs)
    histories = {path.name: path / "history.json" for path in paths}
    return plot_comparison(
        histories, metric=metric, split=split, save=save, **plot_options
    )
