"""Executable Hy evidence checks and offline notebook publication tests."""

import base64
import json
import shutil
import subprocess

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
import pytest

from lispyn.cli import main
from lispyn.compiler import LispynError
from lispyn.evidence import experiment_check, checklist_report
from lispyn.notebook_export import export_markdown
from lispyn.notebooks import build_notebook


def test_evidence_checks_are_explicit_and_json_safe(tmp_path, capsys):
    first = experiment_check(
        "validation improves|baseline", True,
        observed=.82, expected=">= 0.8", notes="synthetic scores",
        save=tmp_path / "first.json",
    )
    second = experiment_check("held-out error", False, observed=.31)
    assert first["passed"] is True
    assert second["passed"] is False
    assert json.loads((tmp_path / "first.json").read_text()) == first
    summary = checklist_report(
        [first, second], title="Synthetic experiment audit",
        save=tmp_path / "audit.json",
    )
    assert summary["passed"] == 1 and summary["failed"] == 1
    assert len(summary["checks"]) == 2
    assert json.loads((tmp_path / "audit.json").read_text()) == summary
    output = capsys.readouterr().out
    assert "[PASS]" in output and "[FAIL]" in output
    assert "Synthetic experiment audit" in output
    assert "validation improves\\|baseline" in output


def test_strict_evidence_failure_still_saves_and_displays(tmp_path, capsys):
    path = tmp_path / "failed.json"
    with pytest.raises(AssertionError, match="Experiment check failed"):
        experiment_check("model quality", False, strict=True, save=path)
    assert json.loads(path.read_text())["passed"] is False
    assert "[FAIL]" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="1 experiment check"):
        checklist_report([{"check": "quality", "passed": False}], strict=True)
    with pytest.raises(TypeError, match="Boolean comparison"):
        experiment_check("ambiguous", 0.9)


@pytest.mark.parametrize("kwargs, error", [
    ({"label": ""}, "label"),
    ({"observed": float("nan")}, "finite"),
    ({"observed": [1, 2]}, "scalar"),
    ({"notes": ""}, "notes"),
    ({"strict": 1}, "strict"),
])
def test_invalid_evidence_inputs(kwargs, error):
    arguments = {"label": "a", "condition": True}
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=error):
        experiment_check(**arguments)


def test_hy_evidence_macros_execute_and_compile(tmp_path):
    pytest.importorskip("hy")
    source = tmp_path / "evidence.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Evaluated scientific claims" "These metrics are synthetic.")\n'
        '(require lispyn.macros [experiment-check checklist-report])\n'
        '(setv metric 0.82)\n'
        '(setv claims [\n'
        '  (experiment-check "validation target" (>= metric 0.8)\n'
        '    :observed metric :expected ">= 0.8")\n'
        '  (experiment-check "no regression" (< metric 0.9))])\n'
        '(setv summary (checklist-report claims :save "evidence.json"))\n'
        '(assert (= (get summary "passed") 2))\n'
        '(print "EVIDENCE MACROS OK")\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    nbformat.validate(notebook)
    assert any("experiment_check" in cell.source
               for cell in notebook.cells if cell.cell_type == "code")
    assert any("checklist_report" in cell.source
               for cell in notebook.cells if cell.cell_type == "code")
    hy = shutil.which("hy")
    assert hy
    completed = subprocess.run(
        [hy, str(source)], capture_output=True, text=True, cwd=tmp_path,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stderr
    assert "EVIDENCE MACROS OK" in completed.stdout
    assert json.loads((tmp_path / "evidence.json").read_text())["passed"] == 2


def _notebook(tmp_path):
    source = tmp_path / "executed.ipynb"
    # A valid 1x1 PNG payload is enough to verify binary extraction.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
    )
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## Hypothesis\nWe expect lower validation loss."),
            new_code_cell(
                "print('success')\n# notebook HTML should never execute",
                execution_count=1,
                outputs=[
                    new_output("stream", name="stdout", text="PASS: 0.82\n"),
                    new_output("display_data", data={
                        "text/markdown": "**Result:** validated",
                        "text/plain": "Result: validated",
                    }),
                    new_output("display_data", data={
                        "image/png": base64.b64encode(png).decode(),
                        "text/plain": "tiny plot",
                    }),
                    new_output("display_data", data={
                        "text/html": "<script>alert('bad')</script>",
                        "text/plain": "safe fallback",
                    }),
                    new_output("display_data", data={
                        "text/html": "<script>alert('omit')</script>",
                    }),
                ],
            ),
            new_code_cell(
                "print('failure example')",
                execution_count=2,
                outputs=[
                    new_output("error", ename="ValueError", evalue="bad",
                               traceback=["\u001b[31mValueError: bad\u001b[0m"]),
                ],
            ),
        ],
        metadata={
            "lispyn": {"source": "experiment.hy", "source_sha256": "123abc"},
        },
    )
    nbformat.write(notebook, source)
    return source, png


def test_export_markdown_preserves_provenance_outputs_and_image(tmp_path, capsys):
    source, png = _notebook(tmp_path)
    target = tmp_path / "report.md"
    assert main(["export", str(source), "-o", str(target)]) == 0
    text = target.read_text()
    assert "**Hy source:** \`experiment.hy\`" in text
    assert "**Hy source SHA-256:** \`123abc\`" in text
    assert "Hypothesis" in text and "validation loss" in text
    assert "\`\`\`python" in text
    assert "print('success')" in text
    assert "PASS: 0.82" in text
    assert "**Result:** validated" in text
    assert "safe fallback" in text
    assert "ValueError: bad" in text
    assert "<script>" not in text and "\u001b" not in text
    assert "![Notebook figure 1](report_assets/figure-001.png)" in text
    assert (tmp_path / "report_assets" / "figure-001.png").read_bytes() == png
    assert "Exported notebook report" in capsys.readouterr().out
    # Export is nonexecuting and must not replace previous reports.
    original = target.read_bytes()
    assert main(["export", str(source), "-o", str(target)]) == 1
    assert target.read_bytes() == original
    assert "overwrite" in capsys.readouterr().err


def test_export_hide_code_without_hiding_results(tmp_path):
    source, _ = _notebook(tmp_path)
    output = tmp_path / "evidence.md"
    assert main(["export", str(source), "-o", str(output), "--hide-code"]) == 0
    text = output.read_text()
    assert "PASS: 0.82" in text
    assert "print('success')" not in text
    assert "Hypothesis" in text
    assert "figure-001.png" in text


def test_export_does_not_execute_untrusted_code(tmp_path):
    file = tmp_path / "no_run.ipynb"
    sentinel = tmp_path / "should_not_exist.txt"
    notebook = new_notebook(cells=[new_code_cell(
        f"open({str(sentinel)!r}, 'w').write('wrong')"
    )])
    nbformat.write(notebook, file)
    report = tmp_path / "no_run.md"
    export_markdown(file, report)
    assert "should_not_exist" in report.read_text()
    assert not sentinel.exists()


@pytest.mark.parametrize("input_name, output_name, error", [
    ("missing.ipynb", "report.md", "existing"),
    ("bad.txt", "report.md", "existing"),
    ("sample.ipynb", "report.txt", ".md"),
])
def test_export_rejects_bad_paths(tmp_path, input_name, output_name, error):
    existing, _ = _notebook(tmp_path)
    src = tmp_path / input_name
    if input_name == "sample.ipynb":
        src = existing
    with pytest.raises(LispynError, match=error):
        export_markdown(src, tmp_path / output_name)


def test_export_rejects_invalid_base64_and_existing_asset_dir(tmp_path):
    source, _ = _notebook(tmp_path)
    file = nbformat.read(source, as_version=4)
    file.cells[1].outputs[2].data["image/png"] = "%%%%not-base64"
    nbformat.write(file, source)
    with pytest.raises(LispynError, match="base64"):
        export_markdown(source, tmp_path / "invalid.md")
    assert not (tmp_path / "invalid.md").exists()
    _notebook(tmp_path)
    (tmp_path / "report_assets").mkdir()
    with pytest.raises(LispynError, match="figure assets"):
        export_markdown(source, tmp_path / "report.md")


def test_export_handles_triple_fences_safely(tmp_path):
    source = tmp_path / "fences.ipynb"
    nb = new_notebook(cells=[new_code_cell('print("a")\n# \`\`\` embedded')])
    nbformat.write(nb, source)
    target = tmp_path / "fences.md"
    export_markdown(source, target)
    assert "\`\`\`\`python" in target.read_text()
