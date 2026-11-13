"""Baseline-aware saved-experiment comparison, including CLI and exports."""

import csv
import json

import pytest

from lispyn.comparison import compare_experiments, format_comparison, save_comparison


def _run(root, name, values, *, metric="loss"):
    directory = root / name
    directory.mkdir()
    (directory / "history.json").write_text(
        json.dumps({"val_" + metric: values}), encoding="utf-8"
    )
    return directory


def test_baseline_gains_for_minimized_loss_and_exports(tmp_path):
    baseline = _run(tmp_path, "baseline", [1.0, 0.8, 0.9])
    candidate = _run(tmp_path, "candidate", [0.9, 0.6, 0.7])
    summary = compare_experiments([baseline, candidate], baseline="baseline")
    assert summary["baseline"] == "baseline"
    assert summary["runs"][0]["best_gain"] == pytest.approx(0)
    assert summary["runs"][1]["best_gain"] == pytest.approx(0.2)
    assert summary["runs"][1]["final_gain"] == pytest.approx(0.2)
    markdown = format_comparison(summary)
    assert "Best gain" in markdown
    assert "\n| baseline |" in markdown
    saved = save_comparison(summary, tmp_path / "comparison.csv")
    with saved.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[1]["best_gain"]) == pytest.approx(0.2)
    assert json.loads(save_comparison(summary, tmp_path / "comparison.json").read_text()) == summary


def test_baseline_gains_for_maximized_accuracy_and_cli(tmp_path, capsys):
    baseline = _run(tmp_path, "baseline", [0.5, 0.7], metric="accuracy")
    candidate = _run(tmp_path, "candidate", [0.6, 0.9], metric="accuracy")
    summary = compare_experiments(
        [baseline, candidate], metric="accuracy", baseline="baseline"
    )
    assert summary["runs"][1]["best_gain"] == pytest.approx(0.2)
    from lispyn.cli import main

    assert main(["compare", str(baseline), str(candidate), "--metric", "accuracy",
                 "--baseline", "baseline"]) == 0
    stdout = capsys.readouterr().out
    assert "Best gain" in stdout
    assert "candidate" in stdout
    with pytest.raises(ValueError, match="Unknown baseline"):
        compare_experiments([baseline, candidate], baseline="not-a-run")
