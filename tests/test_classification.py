"""Classification metrics, confusion-matrix plots and real Hy macro expansion."""

from pathlib import Path
import shutil
import subprocess

import pytest

from lispyn.classification import (
    confusion_counts, normalized_counts, make_confusion_plotter,
)


def test_counts_and_explicit_label_order():
    actual = ["cat", "cat", "dog", "dog", "dog"]
    predicted = ["cat", "dog", "dog", "cat", "dog"]
    counts, classes = confusion_counts(actual, predicted)
    assert classes == ["cat", "dog"]
    assert counts == [[1, 1], [1, 2]]

    reversed_counts, order = confusion_counts(
        actual, predicted, classes=["dog", "cat", "bird"]
    )
    assert order == ["dog", "cat", "bird"]
    assert reversed_counts == [[2, 1, 0], [1, 1, 0], [0, 0, 0]]


def test_normalization_modes_and_absent_classes():
    counts = [[1, 1, 0], [1, 2, 0], [0, 0, 0]]
    assert normalized_counts(counts) == counts
    assert normalized_counts(counts, "true") == [
        [0.5, 0.5, 0.0], [1 / 3, 2 / 3, 0], [0, 0, 0]
    ]
    assert normalized_counts(counts, "pred") == [
        [0.5, 1 / 3, 0], [0.5, 2 / 3, 0], [0, 0, 0]
    ]
    assert normalized_counts(counts, "all")[1][1] == 2 / 5


@pytest.mark.parametrize("actual,predicted,kwargs,match", [
    ([], [], {}, "equal, nonzero"),
    (["a"], ["a", "b"], {}, "equal, nonzero"),
    (["a"], ["missing"], {"classes": ["a"]}, "include every observed"),
    (["a"], ["a"], {"classes": ["a", "a"]}, "unique"),
    (["a"], ["a"], {"classes": [["a"]]}, "hashable"),
    ([["a"]], [["a"]], {}, "hashable"),
])
def test_invalid_classification_data(actual, predicted, kwargs, match):
    with pytest.raises(ValueError, match=match):
        confusion_counts(actual, predicted, **kwargs)


def test_invalid_normalization():
    with pytest.raises(ValueError, match="normalize"):
        normalized_counts([[1]], "incorrect")


def test_dark_confusion_matrix_renders_and_reuses_theme(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    import lispyn.classification as classification
    data = ["cat", "dog", "cat", "bird", "dog", "bird"]
    predictions = ["cat", "cat", "cat", "bird", "dog", "cat"]
    design = make_confusion_plotter(
        theme="docusaurus-dark", palette="turquoise-spectrum",
        normalize="true", classes=["cat", "dog", "bird"],
        title="Original", figsize=(8, 5.5), dpi=160
    )
    fig = design(data, predictions, title="Confusion",
                 save=tmp_path / "cm.png")
    ax = fig.axes[0]
    assert to_rgba(fig.get_facecolor()) == to_rgba("#1B1B1D")
    assert to_rgba(ax.get_facecolor()) == to_rgba("#23272F")
    assert [x.get_text() for x in ax.get_xticklabels()] == [
        "cat", "dog", "bird"
    ]
    assert len(ax.texts) == 9
    assert any(t.get_text() == "1.00" for t in ax.texts)
    assert (tmp_path / "cm.png").stat().st_size > 3000
    plt.close(fig)

    fig = classification.plot_confusion_matrix(
        data, predictions, theme="paper", normalize=None,
        colorbar=False, annotate=False, save=tmp_path / "raw.svg"
    )
    assert len(fig.axes) == 1
    assert not fig.axes[0].texts
    assert (tmp_path / "raw.svg").stat().st_size > 500
    plt.close(fig)


def test_hy_macro_compiles_to_native_python_and_runs(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("hy")
    hy = shutil.which("hy")
    if not hy:
        pytest.skip("Hy executable not available")
    src = tmp_path / "confusion.hy"
    src.write_text(
        '(require lispyn.macros [confusion-matrix defconfusion])\n'
        '(import matplotlib)\n'
        '(matplotlib.use "Agg")\n'
        '(setv actual ["a" "a" "b" "b"])\n'
        '(setv pred ["a" "b" "b" "b"])\n'
        '(defconfusion styled :theme "docusaurus-dark")\n'
        '(setv fig (styled actual pred :classes ["a" "b"] :save "out.png"))\n'
        '(assert (= (len (getattr (get fig.axes 0) "texts")) 4))\n'
        '(setv second (confusion-matrix actual pred :colorbar False))\n'
        '(print "confusion-ok")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [hy, str(src)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "confusion-ok" in result.stdout
    assert (tmp_path / "out.png").stat().st_size > 3000
    from lispyn.notebooks import build_notebook
    nb = build_notebook(src)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "plot_confusion_matrix" in code
    assert "make_confusion_plotter" in code


def test_synthetic_classification_gallery_source_compiles():
    from lispyn.notebooks import build_notebook
    root = Path(__file__).resolve().parents[1]
    nb = build_notebook(root / "examples" / "confusion_matrix.hy")
    assert any("synthetic" in c.source.lower() for c in nb.cells if c.cell_type == "markdown")
    assert any("make_confusion_plotter" in c.source
               for c in nb.cells if c.cell_type == "code")
    assert all(c.get("execution_count") is None
               for c in nb.cells if c.cell_type == "code")
