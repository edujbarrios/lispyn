"""Hy runtime narratives and nonexecuting Jupyter audit regression tests."""

from hashlib import sha256
import json
import shutil
import subprocess

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output
import pytest

from lispyn.cli import main
from lispyn.compiler import LispynError
from lispyn.narrative import (
    render_narrative, render_metric_table, show_narrative,
    show_metric_summary,
)
from lispyn.notebook_audit import audit_notebook
from lispyn.notebooks import build_notebook


def test_render_narrative_with_precision_escaped_braces_and_observations():
    text = render_narrative(
        "## Summary\nAccuracy was {accuracy:.1%}, loss {loss:.2f}, "
        "seed {seed}, selected {selected}; literal {{metric}}.",
        accuracy=.857, loss=.1245, seed=42, selected=True,
    )
    assert "85.7%" in text
    assert "0.12" in text
    assert "seed 42" in text
    assert "selected True" in text
    assert "{metric}" in text


@pytest.mark.parametrize("template, values, problem", [
    ("{missing}", {}, "Missing narrative field"),
    ("{metric.real}", {"metric": 2}, "simple names"),
    ("{metric[0]}", {"metric": 2}, "simple names"),
    ("{metric!r}", {"metric": 2}, "simple names"),
    ("{}", {}, "simple names"),
    ("{metric:.1000000f}", {"metric": 2}, "numeric format"),
    ("{metric:{width}}", {"metric": 2, "width": 10}, "numeric format"),
    ("{metric:.2f}", {"metric": "secret"}, "finite real"),
    ("No placeholders", {"metric": 2}, "Unused"),
    ("{x}", {"x": float("inf")}, "finite"),
    ("{x}", {"x": [1, 2]}, "scalar"),
    ("{x}", {"x": object()}, "scalar"),
    ("{broken", {}, "Invalid narrative template"),
])
def test_template_rejects_missing_names_expressions_bad_types(template, values, problem):
    with pytest.raises(ValueError, match=problem):
        render_narrative(template, **values)


def test_metric_table_is_explicit_and_finite():
    table = render_metric_table(
        {"val_accuracy": .85123, "train_loss|final": .02765},
        title="Evaluation", precision=4,
    )
    assert "### Evaluation" in table
    assert "| val_accuracy | 0.8512 |" in table
    assert "train_loss\\|final" in table
    with pytest.raises(ValueError, match="nonempty mapping"):
        render_metric_table({})
    with pytest.raises(ValueError, match="finite"):
        render_metric_table({"bad": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        render_metric_table({"bool": True})
    with pytest.raises(ValueError, match="precision"):
        render_metric_table({"x": 1}, precision=100)


def test_narratives_display_save_and_retain_markdown(tmp_path, capsys):
    path = tmp_path / "notes" / "finding.md"
    value = show_narrative("**Accuracy:** {accuracy:.0%}", accuracy=.86, save=path)
    assert "**Accuracy:** 86%" == value
    assert path.read_text() == value + "\n"
    assert value in capsys.readouterr().out
    table = show_metric_summary(
        {"val_loss": .25, "val_accuracy": .8},
        title="Scores", precision=3,
        save=tmp_path / "table.md",
    )
    assert "### Scores" in table
    assert "val_accuracy" in (tmp_path / "table.md").read_text()
    with pytest.raises(ValueError, match=".md filename"):
        show_narrative("hello", save=tmp_path / "test.json")


def test_hy_narrative_macros_compile_and_run(tmp_path):
    pytest.importorskip("hy")
    source = tmp_path / "narrative.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Evaluation" "Hypothesis: validation accuracy >= 80%.")\n'
        '(require lispyn.macros [narrate metric-table])\n'
        '(setv accuracy 0.875)\n'
        '(setv loss 0.25)\n'
        '(narrate "Accuracy: {accuracy:.1%}; loss: {loss:.2f}."\n'
        '  :accuracy accuracy :loss loss :save "narrative.md")\n'
        '(metric-table {"val_accuracy" accuracy "val_loss" loss}\n'
        '  :title "Observed scores" :save "metrics.md")\n'
        '(print "NARRATIVES OK")\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells
                     if cell.cell_type == "code")
    assert "show_narrative" in code
    assert "show_metric_summary" in code
    executable = shutil.which("hy")
    assert executable
    run = subprocess.run([executable, str(source)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    assert "NARRATIVES OK" in run.stdout
    assert "87.5%" in (tmp_path / "narrative.md").read_text()
    assert "val_loss" in (tmp_path / "metrics.md").read_text()


def test_audit_covers_clean_executed_notebook_and_true_source_hash(tmp_path, capsys):
    source = tmp_path / "study.hy"
    source.write_text('(print "complete")\n')
    notebook = build_notebook(source)
    assert len([c for c in notebook.cells if c.cell_type == "code"]) == 1
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = 1
            cell.outputs = [new_output("stream", name="stdout", text="complete\n")]
    path = tmp_path / "study.ipynb"
    nbformat.write(notebook, path)
    report = audit_notebook(path, source=source)
    assert report["ready_to_share"] is True
    assert report["source_verified"] is True
    assert report["code_cells"] == report["executed_code_cells"] == 1
    assert report["issues"] == []
    assert report["error_cells"] == 0
    assert report["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert main(["audit", str(path), "--source", str(source)]) == 0
    assert "READY" in capsys.readouterr().out
    assert main(["audit", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source_verified"] is None


def test_audit_detects_missing_execution_errors_and_changed_source(tmp_path):
    source = tmp_path / "study.hy"
    source.write_text('(print "study")\n')
    notebook = build_notebook(source)
    path = tmp_path / "study.ipynb"
    nbformat.write(notebook, path)
    incomplete = audit_notebook(path, source=source)
    assert "unexecuted_code_cells" in incomplete["issues"]
    assert not incomplete["ready_to_share"]

    notebook.cells[-1].execution_count = 1
    notebook.cells[-1].outputs = [
        new_output("error", ename="AssertionError", evalue="failed",
                   traceback=["AssertionError: failed"]),
    ]
    nbformat.write(notebook, path)
    bad = audit_notebook(path, source=source)
    assert bad["error_cells"] == 1
    assert "execution_errors" in bad["issues"]

    source.write_text('(print "changed")\n')
    changed = audit_notebook(path, source=source)
    assert changed["source_verified"] is False
    assert "source_hash_mismatch" in changed["issues"]


def test_audit_with_no_provenance_and_no_code(tmp_path):
    path = tmp_path / "empty.ipynb"
    nbformat.write(new_notebook(cells=[]), path)
    result = audit_notebook(path)
    assert result["ready_to_share"] is False
    assert "no_code_cells" in result["issues"]
    assert "missing_or_invalid_source_sha256" in result["issues"]


def test_audit_never_executes_notebook_code(tmp_path):
    sentinel = tmp_path / "never-created.txt"
    notebook = new_notebook(
        cells=[new_code_cell(
            f'open({str(sentinel)!r}, "w").write("bad")',
            execution_count=1,
        )],
        metadata={"lispyn": {
            "source": "study.hy",
            "source_sha256": "a" * 64,
        }},
    )
    path = tmp_path / "unsafe.ipynb"
    nbformat.write(notebook, path)
    result = audit_notebook(path)
    assert result["ready_to_share"] is True
    assert result["source_verified"] is None
    assert not sentinel.exists()


def test_audit_rejects_bad_paths_and_missing_source(tmp_path):
    with pytest.raises(LispynError, match="existing .ipynb"):
        audit_notebook(tmp_path / "missing.ipynb")
    path = tmp_path / "test.ipynb"
    nbformat.write(new_notebook(cells=[new_code_cell("print(1)")]), path)
    with pytest.raises(LispynError, match="existing .hy"):
        audit_notebook(path, source=tmp_path / "missing.hy")


def test_cli_audit_reports_incomplete_in_json_and_failure_exit(tmp_path, capsys):
    source = tmp_path / "experiment.hy"
    source.write_text('(print 1)\n')
    assert main(["build", str(source)]) == 0
    capsys.readouterr()
    notebook = source.with_suffix(".ipynb")
    assert main(["audit", str(notebook), "--source", str(source), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready_to_share"] is False
    assert report["unexecuted_code_cells"] == 1
    assert report["source_verified"] is True


def test_cli_audit_completed_real_hy_experiment(tmp_path, capsys):
    pytest.importorskip("nbclient")
    source = tmp_path / "executed.hy"
    source.write_text('(print (+ 2 3))\n')
    notebook = tmp_path / "executed.ipynb"
    assert main(["execute", str(source), "-o", str(notebook), "--timeout", "60"]) == 0
    capsys.readouterr()
    assert main(["audit", str(notebook), "--source", str(source), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["code_cells"] == 1
    assert report["executed_code_cells"] == 1
    assert report["error_cells"] == 0
    assert report["ready_to_share"] is True


def test_hy_runtime_narrative_renders_in_jupyter_and_audits(tmp_path, capsys):
    pytest.importorskip("nbclient")
    source = tmp_path / "story.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Measured outcome")\n'
        '(require lispyn.macros [narrate metric-table])\n'
        '(setv score 0.875)\n'
        '(narrate "The observed score was {score:.1%}."\n'
        '  :score score)\n'
        '(metric-table {"observed_score" score} :title "Measured metrics")\n',
        encoding="utf-8",
    )
    output = tmp_path / "story.ipynb"
    assert main(["execute", str(source), "-o", str(output), "--timeout", "60"]) == 0
    capsys.readouterr()
    notebook = nbformat.read(output, as_version=4)
    rich_outputs = [
        str(item.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells if cell.cell_type == "code"
        for item in cell.get("outputs", [])
    ]
    assert any("87.5%" in item for item in rich_outputs)
    assert any("observed_score" in item for item in rich_outputs)
    audit = audit_notebook(output, source=source)
    assert audit["ready_to_share"] is True
    assert audit["source_verified"] is True
