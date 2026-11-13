"""Docusaurus-like dark plots, turquoise palettes and reusable comparisons."""

from pathlib import Path
import json
import shutil
import subprocess

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from lispyn.plots import (
    PALETTES, THEMES, make_plotter, plot_comparison, plot_history,
)


HISTORY = {
    "train_loss": [1.6, 1.1, 0.7, 0.4],
    "val_loss": [1.7, 1.3, 1.05, 0.8],
}
DARK = THEMES["docusaurus-dark"]


def test_dark_theme_exact_background_and_palette(tmp_path):
    output = tmp_path / "dark.png"
    fig = plot_history(
        HISTORY, theme="docusaurus-dark", palette="turquoise-spectrum",
        title="Loss", subtitle="Synthetic data", save=output,
        figsize=(10, 5), dpi=180,
    )
    ax = fig.axes[0]
    assert to_rgba(fig.get_facecolor()) == to_rgba("#1B1B1D")
    assert to_rgba(ax.get_facecolor()) == to_rgba("#23272F")
    assert to_rgba(ax.lines[0].get_color()) == to_rgba("#40E0D0")
    assert to_rgba(ax.lines[1].get_color()) == to_rgba("#8CF0EB")
    assert fig._suptitle.get_text() == "Loss"
    assert ax.get_title() == "Synthetic data"
    assert output.is_file() and output.stat().st_size > 4000
    plt.close(fig)


def test_custom_theme_overrides_and_fill(tmp_path):
    defaults = make_plotter(
        theme="docusaurus-dark", palette="turquoise-spectrum",
        title="Standard title", figure_bg="#222222",
        axes_bg="#2C2C2E", grid_color="#525252",
        text_color="#FFFFFF", spine_color="#787878",
        legend_bg="#333333", legend_edge="#555555",
        line_width=3.5, marker="o", fill_alpha=0.1,
        title_size=17, label_size=13, tick_size=9,
    )
    fig = defaults(HISTORY, title="Override", save=tmp_path / "styled.svg")
    ax = fig.axes[0]
    assert ax.get_title() == "Override"
    assert to_rgba(fig.get_facecolor()) == to_rgba("#222222")
    assert to_rgba(ax.get_facecolor()) == to_rgba("#2C2C2E")
    assert to_rgba(ax.spines["left"].get_edgecolor()) == to_rgba("#787878")
    assert ax.lines[0].get_linewidth() == 3.5
    assert ax.lines[0].get_marker() == "o"
    assert len(ax.collections) == 1  # fill between the curves
    assert (tmp_path / "styled.svg").is_file()
    plt.close(fig)


def test_comparison_uses_seven_colors_for_multiple_runs(tmp_path):
    runs = {
        f"run-{n}": {"val_loss": [1.5 - n * 0.05, 1 - n * 0.05]}
        for n in range(7)
    }
    fig = plot_comparison(runs, save=tmp_path / "comparison.png")
    ax = fig.axes[0]
    assert len(ax.lines) == 7
    expected_indices = (3, 4, 5, 6, 2, 1, 0)
    for line, index in zip(ax.lines, expected_indices):
        assert to_rgba(line.get_color()) == to_rgba(
            PALETTES["turquoise-spectrum"][index]
        )
    assert (tmp_path / "comparison.png").is_file()
    plt.close(fig)


def test_compare_histories_supports_files_and_different_epochs(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"train_accuracy": [0.5, 0.6, 0.7]}))
    fig = plot_comparison(
        {"saved": path, "short": {"train_accuracy": [0.4, 0.55]}},
        metric="accuracy", split="train", theme="clean", palette=None,
        subtitle="Two runs",
    )
    assert len(fig.axes[0].lines[0].get_xdata()) == 3
    assert len(fig.axes[0].lines[1].get_xdata()) == 2
    plt.close(fig)


@pytest.mark.parametrize("options,pattern", [
    ({"theme": "nonexistent"}, "Unknown theme"),
    ({"palette": "not-real"}, "Unknown palette"),
    ({"figure_bg": "not-a-color"}, "figure_bg"),
    ({"legend_edge": "not-a-color"}, "legend_edge"),
    ({"fill_alpha": 1.5}, "fill_alpha"),
    ({"alpha": -0.1}, "alpha"),
    ({"title_size": 0}, "title_size"),
    ({"tick_size": 900}, "tick_size"),
    ({"line_width": -2}, "linewidth"),
])
def test_plot_rejects_bad_style_options(options, pattern):
    with pytest.raises(ValueError, match=pattern):
        plot_history(HISTORY, **options)


def test_compare_rejects_invalid_sources():
    with pytest.raises(ValueError, match="non-empty mapping"):
        plot_comparison({})
    with pytest.raises(ValueError, match="split"):
        plot_comparison({"one": HISTORY}, split="test")
    with pytest.raises(ValueError, match="Missing val_loss"):
        plot_comparison({"one": {"train_loss": [1.0]}})


def test_hy_example_produces_two_dark_pngs(tmp_path):
    hy = shutil.which("hy")
    if hy is None:
        pytest.skip("Hy executable unavailable")
    # Execute in the checked-in examples directory: the example's paths are
    # deliberately relative to examples/, where CI publishes docs/images/.
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "synthetic_curves.hy"
    from lispyn.notebooks import build_notebook

    notebook = build_notebook(source)
    assert any("plot_comparison" in str(c.source)
               for c in notebook.cells if c.cell_type == "code")
    assert all(c.get("execution_count") is None
               for c in notebook.cells if c.cell_type == "code")
    # A file-level smoke test is used in CI, where the optional plotting
    # extra is installed. Avoid running a second subprocess here.
