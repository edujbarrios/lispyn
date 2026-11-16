"""Offline experiment leaderboards, Hy macro and Jupyter execution tests."""

import csv
import json
import shutil
import subprocess

import nbformat
import pytest

from lispyn.cli import main
from lispyn.leaderboard import (
    discover_leaderboard, format_leaderboard, save_leaderboard, show_leaderboard,
)
from lispyn.notebooks import build_notebook


def run(root, name, history, *, status=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "history.json").write_text(json.dumps(history), encoding="utf-8")
    if status is not None:
        (directory / "manifest.json").write_text(
            json.dumps({"status": status, "model": "private"}) + "\n",
            encoding="utf-8",
        )
    return directory


def test_nested_discovery_ranking_and_privacy(tmp_path):
    root = tmp_path / "runs"
    run(root, "baseline", {"val_loss": [0.9, 0.7, 0.8]}, status="completed")
    run(root, "group/tuned", {"val_loss": [0.75, 0.45, 0.5]}, status="completed")
    run(root, "no-manifest", {"val_loss": [0.8, 0.6]}, status=None)
    run(root, "failed", {"val_loss": [0.1]}, status="failed")
    run(root, "no-metric", {"train_loss": [0.3]}, status="completed")
    result = discover_leaderboard(root)
    assert result["metric"] == "loss"
    assert result["split"] == "val"
    assert result["mode"] == "min"
    assert result["rank_by"] == "best"
    assert result["total"] == 3
    assert result["total_candidates"] == 5
    assert [row["run"] for row in result["runs"]] == [
        "group/tuned", "no-manifest", "baseline",
    ]
    assert [row["rank"] for row in result["runs"]] == [1, 2, 3]
    assert result["runs"][0]["best_epoch"] == 2
    assert result["runs"][0]["best"] == 0.45
    assert result["runs"][0]["final"] == 0.5
    assert result["runs"][1]["status"] == "history-only"
    assert result["runs"][2]["status"] == "completed"
    assert len(result["skipped"]) == 2
    as_json = json.dumps(result)
    assert str(root) not in as_json
    assert "private" not in as_json
    assert "### Experiment leaderboard" in format_leaderboard(result)
    assert "Skipped 2" in format_leaderboard(result)


def test_modes_final_metric_and_tie_break(tmp_path):
    root = tmp_path / "runs"
    run(root, "a", {"val_accuracy": [0.60, 0.82, 0.73]})
    run(root, "b", {"val_accuracy": [0.63, 0.78, 0.79]})
    run(root, "c", {"val_accuracy": [0.58, 0.82, 0.75]})
    best = discover_leaderboard(root, metric="accuracy")
    assert best["mode"] == "max"
    assert [x["run"] for x in best["runs"]] == ["a", "c", "b"]
    assert [x["best_epoch"] for x in best["runs"]] == [2, 2, 3]
    final = discover_leaderboard(root, metric="accuracy", rank_by="final", top=2)
    assert [x["run"] for x in final["runs"]] == ["b", "c"]
    assert final["total"] == 3 and len(final["runs"]) == 2
    reversed_mode = discover_leaderboard(root, metric="accuracy", mode="min")
    assert reversed_mode["mode"] == "min"
    assert reversed_mode["runs"][0]["run"] == "c"


def test_export_formats_and_cli_json(tmp_path, capsys):
    root = tmp_path / "runs"
    run(root, "baseline", {"val_loss": [1.2, 0.7]}, status="completed")
    run(root, "tuned", {"val_loss": [1.1, 0.5]}, status="completed")
    report = discover_leaderboard(root)
    md = save_leaderboard(report, tmp_path / "summary.md")
    js = save_leaderboard(report, tmp_path / "summary.json")
    csv_file = save_leaderboard(report, tmp_path / "summary.csv")
    assert "tuned" in md.read_text()
    assert json.loads(js.read_text()) == report
    with csv_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["rank"] == "1" and rows[0]["run"] == "tuned"
    assert "model" not in md.read_text()

    assert main(["leaderboard", str(root), "--top", "1", "--json",
                 "--output", str(tmp_path / "cli.json")]) == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["runs"][0]["run"] == "tuned"
    assert len(parsed["runs"]) == 1
    assert json.loads((tmp_path / "cli.json").read_text()) == parsed
    assert "Leaderboard saved" in captured.err
    assert main(["leaderboard", str(root), "--metric", "loss", "--rank-by", "final"]) == 0
    assert "Experiment leaderboard" in capsys.readouterr().out


def test_strict_rejects_incomplete_and_bad_histories(tmp_path, capsys):
    root = tmp_path / "runs"
    run(root, "ok", {"val_loss": [0.6]})
    run(root, "failed", {"val_loss": [0.1]}, status="running")
    with pytest.raises(ValueError, match="not completed"):
        discover_leaderboard(root, strict=True)
    assert main(["leaderboard", str(root), "--strict"]) == 1
    assert "not completed" in capsys.readouterr().err

    invalid = root / "bad"
    invalid.mkdir()
    (invalid / "history.json").write_text('{"val_loss": [1, true]}')
    result = discover_leaderboard(root)
    assert any(item["run"] == "bad" for item in result["skipped"])
    with pytest.raises(ValueError, match="invalid val_loss"):
        discover_leaderboard(root, strict=True)


@pytest.mark.parametrize("options,error", [
    ({"metric": "val_loss"}, "unprefixed"),
    ({"metric": ""}, "unprefixed"),
    ({"split": "test"}, "split"),
    ({"mode": "least"}, "mode"),
    ({"rank_by": "epoch"}, "rank_by"),
    ({"top": 0}, "top"),
    ({"top": 101}, "top"),
    ({"top": True}, "top"),
    ({"strict": 1}, "Boolean"),
])
def test_invalid_leaderboard_options(tmp_path, options, error):
    run(tmp_path, "ok", {"val_loss": [0.5]})
    with pytest.raises((ValueError, TypeError), match=error):
        discover_leaderboard(tmp_path, **options)


def test_refuses_empty_directory_and_invalid_output(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        discover_leaderboard(tmp_path / "absent")
    with pytest.raises(ValueError, match="No history.json"):
        discover_leaderboard(tmp_path)
    with pytest.raises(ValueError, match="Leaderboard output"):
        save_leaderboard({"runs": []}, tmp_path / "invalid.txt")


def test_markdown_escapes_untrusted_names(tmp_path):
    root = tmp_path / "runs"
    run(root, "name|with<piece>", {"val_loss": [0.8]})
    text = format_leaderboard(discover_leaderboard(root))
    assert "name\\|with&lt;piece&gt;" in text
    assert "<piece>" not in text


def test_terminal_hy_macro_compiles_and_executes(tmp_path):
    pytest.importorskip("hy")
    root = tmp_path / "runs"
    run(root, "baseline", {"val_loss": [1.0, 0.75]})
    run(root, "tuned", {"val_loss": [0.9, 0.6]})
    source = tmp_path / "leaderboard.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Compare experiment artifacts")\n'
        '(require lispyn.macros [experiment-leaderboard])\n'
        '(setv board (experiment-leaderboard "runs"\n'
        '  :metric "loss" :rank-by "best" :save "leaderboard.md"))\n'
        '(assert (= (get (get (get board "runs") 0) "run") "tuned"))\n'
        '(print "LEADERBOARD MACRO PASSED")\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    assert any("show_leaderboard" in cell.source
               for cell in notebook.cells if cell.cell_type == "code")
    executable = shutil.which("hy")
    assert executable
    run_result = subprocess.run(
        [executable, str(source)], cwd=tmp_path,
        capture_output=True, text=True, timeout=45,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "LEADERBOARD MACRO PASSED" in run_result.stdout
    assert "tuned" in (tmp_path / "leaderboard.md").read_text()


def test_leaderboard_macro_renders_in_jupyter(tmp_path, capsys):
    pytest.importorskip("nbclient")
    root = tmp_path / "runs"
    run(root, "baseline", {"val_loss": [1.0, 0.75]})
    run(root, "tuned", {"val_loss": [0.9, 0.6]})
    source = tmp_path / "leaderboard.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Aggregate experiment outcomes")\n'
        '(require lispyn.macros [experiment-leaderboard])\n'
        '(experiment-leaderboard "runs" :metric "loss")\n',
        encoding="utf-8",
    )
    assert main(["execute", str(source), "--timeout", "60"]) == 0
    capsys.readouterr()
    notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
    markdown = [
        str(item.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells if cell.cell_type == "code"
        for item in cell.get("outputs", [])
    ]
    assert any("Experiment leaderboard" in text and "tuned" in text
               for text in markdown)
