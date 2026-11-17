"""Class balance reports and plots work in Hy scripts and Jupyter notebooks."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "label_balance.hy"


def test_balance_report_shows_counts_and_ratios(tmp_path, capsys):
    from lispyn.label_balance import label_balance
    labels = ["cat", "cat", "cat", "dog", "bird", "dog"]
    original = labels.copy()
    output = tmp_path / "balance.json"
    report = label_balance(labels, title="Animals", save=output)
    assert report == json.loads(output.read_text())
    assert report["total"] == 6 and report["num_classes"] == 3
    assert report["classes"] == [
        {"label": "cat", "count": 3, "share": 0.5},
        {"label": "dog", "count": 2, "share": 2 / 6},
        {"label": "bird", "count": 1, "share": 1 / 6},
    ]
    assert report["majority_share"] == 0.5
    assert report["imbalance_ratio"] == 3
    assert labels == original
    assert "Animals" in capsys.readouterr().out
    saved = tmp_path / "balance.md"
    label_balance(["<b>cat</b>", "a|b"], save=saved)
    assert "&lt;b&gt;cat&lt;/b&gt;" in saved.read_text()
    assert r"a\|b" in saved.read_text()


def test_balance_accepts_pandas_numpy_and_torch():
    from lispyn.label_balance import label_balance
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    for values in (np.array([0, 0, 1]), pd.Series(["a", "a", "b"])):
        result = label_balance(values)
        assert result["total"] == 3 and result["num_classes"] == 2
    try:
        import torch
    except ImportError:
        return
    result = label_balance(torch.tensor([0, 0, 1]))
    assert result["num_classes"] == 2 and result["classes"][0]["count"] == 2


@pytest.mark.parametrize("labels,kwargs,exc,match", [
    ([], {}, ValueError, "empty"),
    ("dogs", {}, TypeError, "sequence"),
    ({"a": 2}, {}, TypeError, "sequence"),
    ([[1, 2]], {}, TypeError, "scalar"),
    ([1, float("nan")], {}, ValueError, "NaN"),
    ([1, float("inf")], {}, ValueError, "infinity"),
    ([1, "1"], {}, ValueError, "identical"),
    (["", "cat"], {}, ValueError, "nonempty"),
    (["a", "b", "c"], {"max_classes": 2}, ValueError, "distinct classes"),
    (["x"], {"max_classes": True}, ValueError, "max_classes"),
    (["x"], {"title": ""}, ValueError, "title"),
])
def test_invalid_balance_data(labels, kwargs, exc, match):
    from lispyn.label_balance import label_balance
    with pytest.raises(exc, match=match):
        label_balance(labels, **kwargs)



def test_class_weights_align_with_encoded_class_order_and_preserve_input():
    from lispyn.label_balance import balanced_class_weights
    labels = ["cat", "cat", "cat", "dog", "dog", "bird"]
    original = labels.copy()
    result = balanced_class_weights(labels, class_order=["bird", "cat", "dog"])
    assert result["class_order"] == ["bird", "cat", "dog"]
    assert result["counts"] == [1, 3, 2]
    assert result["weights"] == pytest.approx([2.0, 2 / 3, 1.0])
    assert result["total"] == 6 and result["num_classes"] == 3
    assert labels == original
    assert balanced_class_weights(labels)["class_order"] == ["cat", "dog", "bird"]


@pytest.mark.parametrize("order", [
    ["cat", "dog"],                     # missing observed class
    ["cat", "dog", "bird", "fox"],       # unexpected class
    ["cat", "cat", "bird"],             # duplicate and missing class
    ["cat", "dog", "bird", "bird"],    # duplicate
])
def test_class_weights_reject_invalid_class_order(order):
    from lispyn.label_balance import balanced_class_weights
    with pytest.raises(ValueError, match="class_order"):
        balanced_class_weights(["cat", "dog", "bird"], class_order=order)


def test_class_weights_require_matching_label_types():
    from lispyn.label_balance import balanced_class_weights
    with pytest.raises(ValueError, match="class_order"):
        balanced_class_weights([1, 1, 2], class_order=["1", 2])
    with pytest.raises(ValueError, match="class_order"):
        balanced_class_weights([True, False, False], class_order=[1, 0])
    assert balanced_class_weights([1, 1, 2], class_order=[2, 1])["counts"] == [1, 2]
    with pytest.raises(ValueError, match="distinct classes"):
        balanced_class_weights(["a", "b", "c"], max_classes=2)


def test_class_weights_numpy_and_pandas():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from lispyn.label_balance import balanced_class_weights
    for labels in (np.array([0, 1, 0, 2]), pd.Series([0, 1, 0, 2])):
        result = balanced_class_weights(labels, class_order=[0, 1, 2])
        assert result["counts"] == [2, 1, 1]
        assert result["weights"] == pytest.approx([2 / 3, 4 / 3, 4 / 3])


def test_plot_is_real_matplotlib_figure_and_saves_png_svg(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from lispyn.label_balance import plot_label_balance

    for extension in (".png", ".svg"):
        destination = tmp_path / ("labels" + extension)
        figure = plot_label_balance(["cat", "dog", "cat"],
                                    title="Model labels", theme="paper",
                                    save=destination)
        assert isinstance(figure, Figure)
        assert len(figure.axes[0].patches) == 2
        assert destination.stat().st_size > 200
        plt.close(figure)
    with pytest.raises(ValueError, match="plot must be"):
        plot_label_balance(["cat"], save=tmp_path / "bad.txt")
    with pytest.raises(ValueError, match="figsize"):
        plot_label_balance(["cat"], figsize=[0, 1])


def test_hy_demo_run_build_and_execute(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("nbclient")
    import nbformat
    from lispyn.notebooks import build_notebook

    notebook = build_notebook(EXAMPLE)
    nbformat.validate(notebook)
    assert any(c.cell_type == "markdown" for c in notebook.cells)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "lispyn.label_balance.label_balance" in code
    assert "lispyn.label_balance.plot_label_balance" in code
    assert "lispyn.label_balance.balanced_class_weights" in code

    cli = [sys.executable, "-m", "lispyn.cli"]
    direct = subprocess.run(cli + ["run", str(EXAMPLE)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 0, direct.stderr
    assert "Class balance checks passed!" in direct.stdout

    template_path = tmp_path / "template.ipynb"
    built = subprocess.run(cli + ["build", str(EXAMPLE), "-o", str(template_path)],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    template = nbformat.read(template_path, as_version=4)
    assert all(c.execution_count is None and not c.outputs
               for c in template.cells if c.cell_type == "code")

    executed_path = tmp_path / "executed.ipynb"
    run = subprocess.run(cli + ["execute", str(EXAMPLE), "-o", str(executed_path),
                                "--timeout", "90"],
                         cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    executed = nbformat.read(executed_path, as_version=4)
    assert all(c.execution_count is not None for c in executed.cells
               if c.cell_type == "code")
    texts = [str(o.get("text", "")) for c in executed.cells
             if c.cell_type == "code" for o in c.get("outputs", [])]
    assert "Class balance checks passed!" in "\n".join(texts)
