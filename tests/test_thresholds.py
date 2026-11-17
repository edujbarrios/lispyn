"""Binary threshold metrics: pure Python, NumPy/pandas and Hy/Jupyter regression tests."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "binary_thresholds.hy"


def test_threshold_sweep_confusion_metrics_and_best_selection():
    from lispyn.thresholds import sweep_binary_thresholds
    labels = [0, 0, 1, 1, 1, 0, 1, 0]
    scores = [0.1, 0.35, 0.4, 0.8, 0.7, 0.9, 0.55, 0.2]
    original = scores.copy()
    report = sweep_binary_thresholds(labels, scores, thresholds=[0.75, 0.3, 0.5])
    assert scores == original
    assert report["samples"] == 8
    assert report["positives"] == report["negatives"] == 4
    assert report["best_threshold"] == 0.3
    assert [row["threshold"] for row in report["rows"]] == [0.3, 0.5, 0.75]
    low, middle, high = report["rows"]
    assert (low["tp"], low["fp"], low["fn"], low["tn"]) == (4, 2, 0, 2)
    assert low["precision"] == pytest.approx(2 / 3)
    assert low["recall"] == 1.0
    assert low["f1"] == pytest.approx(0.8)
    assert low["balanced_accuracy"] == pytest.approx(0.75)
    assert (middle["tp"], middle["fp"], middle["fn"], middle["tn"]) == (3, 1, 1, 3)
    assert high["f1"] == pytest.approx(1 / 3)
    assert report["best_metrics"] == low


def test_balanced_accuracy_objective_ties_and_zero_division():
    from lispyn.thresholds import sweep_binary_thresholds
    report = sweep_binary_thresholds([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8],
                                     optimize="balanced_accuracy")
    assert report["best_threshold"] == 0.5  # closest to 0.5 in a perfect tie
    assert report["best_metrics"]["balanced_accuracy"] == 1.0
    edge = sweep_binary_thresholds([0, 1], [0.2, 0.8], thresholds=[0.0, 1.0])
    assert edge["rows"][0]["precision"] == pytest.approx(0.5)
    assert edge["rows"][1]["precision"] == 0.0
    assert edge["rows"][1]["recall"] == 0.0
    assert edge["rows"][1]["f1"] == 0.0
    assert edge["rows"][1]["specificity"] == 1.0


@pytest.mark.parametrize("labels,scores,kwargs,match", [
    ([], [], {}, "between 1"),
    ([0, 1], [0.3], {}, "equal lengths"),
    ([0, 0], [0.1, 0.5], {}, "both positive and negative"),
    ([0, 1.0], [0.1, 0.5], {}, "integer labels"),
    ([0, 2], [0.1, 0.5], {}, "integer labels"),
    ([0, 1], [True, 0.5], {}, "finite probabilities"),
    ([0, 1], [float("nan"), 0.5], {}, "finite probabilities"),
    ([0, 1], [float("inf"), 0.5], {}, "finite probabilities"),
    ([0, 1], [0.5, 1.5], {}, "finite probabilities"),
    ([0, 1], [0.3, 0.5], {"thresholds": [0.5, 0.5]}, "unique"),
    ([0, 1], [0.3, 0.5], {"thresholds": [float("nan")]}, "thresholds"),
    ([0, 1], [0.3, 0.5], {"thresholds": [True]}, "thresholds"),
    ([0, 1], [0.3, 0.5], {"thresholds": [1.1]}, "thresholds"),
    ([0, 1], [0.3, 0.5], {"thresholds": []}, "between 1"),
    ([0, 1], [0.3, 0.5], {"thresholds": [i / 110 for i in range(110)]}, "101"),
    ([0, 1], [0.3, 0.5], {"optimize": "auc"}, "optimize"),
    ([0, 1], [0.3, 0.5], {"title": ""}, "title"),
])
def test_rejects_invalid_inputs(labels, scores, kwargs, match):
    from lispyn.thresholds import sweep_binary_thresholds
    with pytest.raises((TypeError, ValueError), match=match):
        sweep_binary_thresholds(labels, scores, **kwargs)


def test_numpy_pandas_inputs_and_one_dimensional_validation():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from lispyn.thresholds import sweep_binary_thresholds
    result = sweep_binary_thresholds(np.array([0, 1, 0, 1]),
                                     pd.Series([0.1, 0.6, 0.9, 0.7]),
                                     thresholds=np.array([0.5]))
    assert result["rows"][0]["tp"] == 2 and result["rows"][0]["fp"] == 1
    with pytest.raises(ValueError, match="one-dimensional"):
        sweep_binary_thresholds(np.array([[0, 1]]), [0.1, 0.8])


def test_markdown_and_json_exports_are_aggregate_only(tmp_path, capsys):
    from lispyn.thresholds import threshold_report
    labels, scores = [0, 1, 1, 0], [0.0123456789, 0.8, 0.7, 0.98]
    json_path = tmp_path / "metrics.json"
    report = threshold_report(labels, scores, thresholds=[0.2, 0.5], save=json_path,
                              title="<validation|only>")
    assert report == json.loads(json_path.read_text(encoding="utf-8"))
    assert "0.0123456789" not in json_path.read_text(encoding="utf-8")
    assert "<validation|only>" in report["title"]
    md_path = tmp_path / "metrics.md"
    threshold_report(labels, scores, thresholds=[0.2, 0.5], save=md_path)
    markdown = md_path.read_text(encoding="utf-8")
    assert "| Threshold | TP | FP | FN | TN |" in markdown
    assert "0.0123456789" not in markdown
    assert "validation" in capsys.readouterr().out
    with pytest.raises(ValueError, match="md or .json"):
        threshold_report(labels, scores, save=tmp_path / "bad.csv")


def test_threshold_plot_exports_real_figures(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from lispyn.thresholds import plot_threshold_metrics
    for extension in (".svg", ".png"):
        output = tmp_path / ("thresholds" + extension)
        figure = plot_threshold_metrics(
            [0, 1, 0, 1], [0.2, 0.6, 0.1, 0.9],
            thresholds=[0.2, 0.5, 0.8], save=output
        )
        assert isinstance(figure, Figure)
        assert len(figure.axes[0].lines) == 5
        assert output.stat().st_size > 200
        plt.close(figure)
    with pytest.raises(ValueError, match="threshold plot"):
        plot_threshold_metrics([0, 1], [0.1, 0.9], save=tmp_path / "bad.txt")
    with pytest.raises(ValueError, match="figsize"):
        plot_threshold_metrics([0, 1], [0.1, 0.9], figsize=[1, 0])


def test_hy_example_run_build_and_execute(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("nbclient")
    import nbformat
    from lispyn.notebooks import build_notebook
    notebook = build_notebook(EXAMPLE)
    nbformat.validate(notebook)
    assert any(c.cell_type == "markdown" for c in notebook.cells)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "lispyn.thresholds.threshold_report" in code
    assert "lispyn.thresholds.plot_threshold_metrics" in code
    command = [sys.executable, "-m", "lispyn.cli"]
    direct = subprocess.run(command + ["run", str(EXAMPLE)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 0, direct.stderr
    assert "Threshold diagnostics passed!" in direct.stdout

    target = tmp_path / "template.ipynb"
    built = subprocess.run(command + ["build", str(EXAMPLE), "-o", str(target)],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    unexecuted = nbformat.read(target, as_version=4)
    assert all(cell.execution_count is None and not cell.outputs
               for cell in unexecuted.cells if cell.cell_type == "code")

    target = tmp_path / "executed.ipynb"
    run = subprocess.run(command + [
        "execute", str(EXAMPLE), "-o", str(target), "--timeout", "90"
    ], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    executed = nbformat.read(target, as_version=4)
    assert all(cell.execution_count is not None
               for cell in executed.cells if cell.cell_type == "code")
    output_text = "\n".join(
        str(output.get("text", "")) for cell in executed.cells
        if cell.cell_type == "code" for output in cell.get("outputs", [])
    )
    assert "Threshold diagnostics passed!" in output_text
