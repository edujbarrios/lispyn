"""Offline plotting tests, with no downloaded datasets or models."""

import json

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from lispyn.plots import plot_history


def test_plot_loss_and_accuracy_from_history(tmp_path):
    history = {
        "train_loss": [0.9, 0.6, 0.4],
        "val_loss": [1.0, 0.8, 0.7],
        "train_accuracy": [0.5, 0.75, 1.0],
        "val_accuracy": [0.25, 0.5, 0.75],
    }
    loss_path = tmp_path / "plots" / "loss.png"
    fig = plot_history(history, save=loss_path)
    assert loss_path.is_file() and loss_path.stat().st_size > 100
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert [float(x) for x in ax.lines[0].get_ydata()] == history["train_loss"]
    plt.close(fig)

    json_path = tmp_path / "history.json"
    json_path.write_text(json.dumps(history))
    fig = plot_history(json_path, metric="accuracy", save=tmp_path / "acc.svg")
    assert (tmp_path / "acc.svg").is_file()
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_plot_run_object_and_training_only(tmp_path):
    from types import SimpleNamespace
    result = SimpleNamespace(history={"train_loss": [1, 0.5], "val_loss": []})
    fig = plot_history(result)
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)


def test_reject_bad_metric_and_values():
    with pytest.raises(ValueError, match="unprefixed"):
        plot_history({"train_loss": [1]}, metric="train_loss")
    with pytest.raises(ValueError, match="No recorded"):
        plot_history({"train_loss": []})
    with pytest.raises(ValueError, match="non-numeric"):
        plot_history({"train_loss": [True]})
    with pytest.raises(ValueError, match="non-finite"):
        plot_history({"train_loss": [float("nan")]})
    with pytest.raises(TypeError, match="Run object"):
        plot_history(42)


def test_cli_report_can_save_a_figure(tmp_path, capsys):
    from lispyn.cli import main

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "model": "Example", "device": "cpu",
        "config": {"epochs": 2, "tuning": "full", "seed": 1},
        "summary": {"monitor": "val_loss", "value": 0.4, "epoch": 2},
    }))
    (run / "history.json").write_text(json.dumps({
        "train_loss": [1.0, 0.8], "val_loss": [0.6, 0.4]
    }))
    plot_path = run / "chart.png"
    assert main(["report", str(run), "--plot", str(plot_path)]) == 0
    assert plot_path.is_file()
    assert "Plot saved:" in capsys.readouterr().out
