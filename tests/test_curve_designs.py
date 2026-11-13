"""Offline plots, theme reuse, and true Hy macro compilation."""

from pathlib import Path
import shutil
import subprocess

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from lispyn.plots import THEMES, make_plotter, plot_history


HISTORY = {
    "train_loss": [1.1, 0.75, 0.52],
    "val_loss": [1.2, 0.85, 0.69],
    "train_accuracy": [0.3, 0.6, 0.8],
    "val_accuracy": [0.25, 0.5, 0.7],
}


def test_all_themes_render_without_modifying_matplotlib_global_style():
    previous = plt.rcParams["axes.facecolor"]
    for theme in THEMES:
        fig = plot_history(HISTORY, theme=theme)
        assert to_rgba(fig.get_facecolor()) == to_rgba(THEMES[theme]["background"])
        assert to_rgba(fig.axes[0].get_facecolor()) == to_rgba(THEMES[theme]["axes"])
        assert len(fig.axes[0].lines) == 2
        plt.close(fig)
    assert plt.rcParams["axes.facecolor"] == previous


def test_custom_colors_text_size_and_export(tmp_path):
    out = tmp_path / "plots" / "training.svg"
    fig = plot_history(
        HISTORY, theme="dark", metric="accuracy", save=out,
        title="Precisión del experimento", xlabel="Época", ylabel="Aciertos",
        train_label="Ajuste", val_label="Validación",
        train_color="#22D3EE", val_color="gold",
        figsize=(8, 3), linewidth=3, grid=False, legend=True,
    )
    ax = fig.axes[0]
    assert ax.get_title() == "Precisión del experimento"
    assert ax.get_xlabel() == "Época"
    assert ax.get_ylabel() == "Aciertos"
    assert tuple(fig.get_size_inches()) == (8.0, 3.0)
    assert to_rgba(ax.lines[0].get_color()) == to_rgba("#22D3EE")
    assert to_rgba(ax.lines[1].get_color()) == to_rgba("gold")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Ajuste", "Validación"]
    assert out.exists() and out.stat().st_size > 200
    plt.close(fig)


def test_reusable_design_and_per_call_overrides(tmp_path):
    design = make_plotter(theme="paper", title="Base", train_color="#AABBCC")
    fig = design(HISTORY)
    assert fig.axes[0].get_title() == "Base"
    assert to_rgba(fig.axes[0].lines[0].get_color()) == to_rgba("#AABBCC")
    plt.close(fig)
    fig = design(HISTORY, metric="accuracy", title="Override",
                 save=tmp_path / "accuracy.png")
    assert fig.axes[0].get_title() == "Override"
    assert fig.axes[0].get_ylabel() == "Accuracy"
    assert (tmp_path / "accuracy.png").is_file()
    plt.close(fig)


def test_invalid_themes_colors_and_dimensions():
    with pytest.raises(ValueError, match="Unknown theme"):
        plot_history(HISTORY, theme="unknown")
    with pytest.raises(ValueError, match="train_color"):
        plot_history(HISTORY, train_color="not-a-color")
    with pytest.raises(ValueError, match="val_color"):
        plot_history(HISTORY, val_color="invalid-color")
    with pytest.raises(ValueError, match="figsize"):
        plot_history(HISTORY, figsize=(0, 4))
    with pytest.raises(ValueError, match="linewidth"):
        plot_history(HISTORY, linewidth=-1)
    with pytest.raises(ValueError, match="dpi"):
        plot_history(HISTORY, dpi=0)


def test_hy_macros_expand_and_execute(tmp_path):
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook

    file = tmp_path / "styles.hy"
    file.write_text(
        '(require lispyn.macros [curve defcurve])\n'
        '(import matplotlib.pyplot :as plt)\n'
        '(setv h {"train_loss" [1.0 0.5] "val_loss" [1.1 0.6]})\n'
        '(defcurve shared :theme "paper" :train-color "#336699")\n'
        '(setv first (shared h :title "My chart" :save "first.png"))\n'
        '(assert (= (.get_title (get first.axes 0)) "My chart"))\n'
        '(.close plt first)\n'
        '(setv second (curve h :theme "dark" :val-color "#ff8800" :save "second.svg"))\n'
        '(.close plt second)\n'
        '(print "macro-curve-ok")\n',
        encoding="utf-8",
    )
    hy = shutil.which("hy")
    assert hy, "Hy executable required for macro integration test"
    result = subprocess.run([hy, str(file)], cwd=tmp_path, text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "macro-curve-ok" in result.stdout
    assert (tmp_path / "first.png").stat().st_size > 200
    assert (tmp_path / "second.svg").stat().st_size > 200
    nb = build_notebook(file)
    assert "make_plotter" in nb.cells[-1].source
    assert "plot_history" in nb.cells[-1].source


def test_example_compiles_multicell_styles():
    from lispyn.notebooks import build_notebook

    src = Path(__file__).resolve().parents[1] / "examples" / "curve_designs.hy"
    nb = build_notebook(src)
    assert "make_plotter" in nb.cells[-1].source
    assert "plot_history" in nb.cells[-1].source
