"""Notebook parameter tags and human-readable experiment results in Hy."""

from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess

import nbformat
import pytest

from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook, split_markdown_macros
from lispyn.reporting import render_run_report, show_run_report


def test_experiment_parameters_are_tagged_and_compiled(tmp_path):
    source = tmp_path / "parameters.hy"
    source.write_text(
        '(require lispyn.macros [md experiment-params])\n'
        '(md "# Reproducible experiment")\n'
        '(require lispyn.macros [experiment-params])\n'
        '(experiment-params\n'
        '  (setv seed 42)\n'
        '  (setv epochs 3)\n'
        '  (setv lr 0.01))\n'
        '(print seed epochs lr)\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    parameter_cells = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1
    assert "seed = 42" in parameter_cells[0].source
    assert "epochs = 3" in parameter_cells[0].source
    assert "lr = 0.01" in parameter_cells[0].source
    assert "experiment_params(" not in parameter_cells[0].source
    assert "parameters" not in notebook.cells[-1].metadata.get("tags", [])
    assert [c.id for c in notebook.cells] == [c.id for c in build_notebook(source).cells]


def test_parameter_cell_accepts_optional_hide_input_tag(tmp_path):
    source = tmp_path / "tagged.hy"
    source.write_text(
        '(require lispyn.macros [cell-tags experiment-params])\n'
        '(cell-tags "hide-input")\n'
        '(experiment-params\n'
        '  (setv epochs 10)\n'
        '  (setv batch-size 8))\n'
    )
    nb = build_notebook(source)
    parameter_cells = [cell for cell in nb.cells
                       if "parameters" in cell.metadata.get("tags", [])]
    assert len(parameter_cells) == 1
    assert parameter_cells[0].metadata["tags"] == ["parameters", "hide-input"]
    assert "batch_size = 8" in parameter_cells[0].source


@pytest.mark.parametrize("source, message", [
    ('(experiment-params)\n', "one or more"),
    ('(experiment-params seed 42)\n', "setv name value"),
    ('(experiment-params (print 42))\n', "setv name value"),
    ('(experiment-params (setv seed 42 43))\n', "setv name value"),
    ('(do\n  (experiment-params (setv seed 1)))\n', "top-level"),
    ('(experiment-params (setv seed 42)) (print seed)\n', "own lines"),
])
def test_malformed_parameters_rejected(source, message, tmp_path):
    file = tmp_path / "bad.hy"
    file.write_text(source)
    with pytest.raises(LispynError, match=message):
        build_notebook(file)


def test_parameters_and_report_macro_execute_in_hy(tmp_path):
    pytest.importorskip("hy")
    source = tmp_path / "story.hy"
    source.write_text(
        '(require lispyn.macros [experiment-params report-run])\n'
        '(experiment-params (setv seed 42) (setv epochs 4))\n'
        '(assert (= seed 42))\n'
        '(setv history {"train_loss" [1.2 0.8 0.5 0.4]\n'
        '               "val_loss" [1.4 0.9 0.6 0.7]})\n'
        '(report-run history :title "My Hy experiment"\n'
        '  :notes "Synthetic metric histories" :save "report.md")\n'
        '(print "STORY OK" epochs)\n'
    )
    executable = shutil.which("hy")
    assert executable, "Hy CLI not available"
    run = subprocess.run([executable, str(source)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=40)
    assert run.returncode == 0, run.stderr
    assert "STORY OK 4" in run.stdout
    report = (tmp_path / "report.md").read_text()
    assert "My Hy experiment" in report
    assert "Synthetic metric histories" in report
    assert "| val_loss | min | 3 | 0.6 | 0.7 |" in report
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    assert any("parameters" in c.metadata.get("tags", []) for c in notebook.cells)
    assert any("show_run_report" in c.source for c in notebook.cells
               if c.cell_type == "code")


def test_run_report_infers_directions_epochs_and_early_stop():
    run = SimpleNamespace(
        history={
            "train_loss": [1.5, 0.9, 0.6],
            "val_loss": [1.4, 0.7, 0.8],
            "val_accuracy": [0.4, 0.8, 0.75],
        },
        stopped_early=True,
    )
    output = render_run_report(
        run, title="ResNet|Study", notes="Validated with a held-out split"
    )
    assert "**Epochs recorded:** 3" in output
    assert "**Early stopping:** yes" in output
    assert "| val_loss | min | 2 | 0.7 | 0.8 |" in output
    assert "| val_accuracy | max | 2 | 0.8 | 0.75 |" in output
    assert "## Notes" in output
    assert "Dataset" not in output
    assert "ResNet|Study" in output
    selected = render_run_report(run, metric="val_accuracy", mode="min")
    assert "| val_accuracy | min | 1 | 0.4 | 0.75 |" in selected
    assert "| val_loss |" not in selected


def test_report_save_and_terminal_output(tmp_path, capsys):
    history = {"train_loss": [3, 2, 1], "val_loss": [2, 1, 2]}
    path = tmp_path / "reports" / "experiment.md"
    result = show_run_report(history, title="Results", save=path)
    assert path.read_text() == result
    assert "# Results" in capsys.readouterr().out
    with pytest.raises(ValueError, match=".md extension"):
        show_run_report(history, save=tmp_path / "report.json")


@pytest.mark.parametrize("history, error", [
    ({}, "No training"),
    ({"val_loss": []}, "cannot be empty"),
    ({"val_loss": ["wrong"]}, "finite real"),
    ({"train_loss": [float("nan")]}, "finite real"),
    ({"train_loss": [True]}, "finite real"),
])
def test_invalid_report_history(history, error):
    with pytest.raises(ValueError, match=error):
        render_run_report(history)


def test_report_validates_title_mode_metric():
    history = {"val_loss": [1, 2]}
    with pytest.raises(ValueError, match="title"):
        render_run_report(history, title=" ")
    with pytest.raises(ValueError, match="mode"):
        render_run_report(history, mode="largest")
    with pytest.raises(ValueError, match="absent"):
        render_run_report(history, metric="val_accuracy")
    with pytest.raises(TypeError, match="notes"):
        render_run_report(history, notes=123)
    with pytest.raises(TypeError, match="mapping"):
        render_run_report("wrong")
