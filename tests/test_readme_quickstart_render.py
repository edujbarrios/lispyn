"""README quickstart snapshots must come from executed notebook evidence."""
from hashlib import sha256
import importlib.util
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_readme_quickstart", ROOT / "scripts" / "render_readme_quickstart.py"
)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def _sample(tmp_path, *, executed=True, error=False):
    source = tmp_path / "study.hy"
    source.write_text('(print "a real experiment")\n', encoding="utf-8")
    code = new_code_cell(
        "from lispyn.plots import plot_history\n"
        "plot_history(run, metric='loss')",
        execution_count=1 if executed else None,
        outputs=[new_output("stream", name="stdout", text="measured: 0.42\n")]
                if executed else [],
    )
    if error:
        code.outputs.append(new_output(
            "error", ename="RuntimeError", evalue="oops",
            traceback=["RuntimeError: oops"],
        ))
    notebook = new_notebook(
        cells=[new_markdown_cell("## The experiment"), code],
        metadata={"lispyn": {
            "source": source.name,
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
        }},
    )
    path = tmp_path / "executed.ipynb"
    nbformat.write(notebook, path)
    return source, path


def test_renderer_includes_every_cell_and_literal_output(tmp_path):
    source, notebook = _sample(tmp_path)
    figure = ROOT / "example_generated_notebooks" / "assets" / "synthetic_training_curves.png"
    section = renderer.render(source, notebook, figure)
    assert "Cell 1 · Markdown" in section
    assert "Cell 2 · Code · executed #1" in section
    assert "measured: 0.42" in section
    assert "plot_history(run" in section
    assert "quickstart_training_curves.png" in section
    assert "## The experiment" in section

    readme = tmp_path / "README.md"
    readme.write_text(f"Intro\n{renderer.BEGIN}\nplaceholder\n{renderer.END}\nEnd\n")
    renderer.rewrite(readme, section)
    renderer.rewrite(readme, section, check=True)
    assert section in readme.read_text()
    assert readme.read_text().count("measured: 0.42") == 1
    with pytest.raises(ValueError, match="stale"):
        renderer.rewrite(readme, section + "unverified extra text", check=True)


@pytest.mark.parametrize("executed,error", [
    (False, False),
    (True, True),
])
def test_renderer_refuses_missing_execution_or_errors(tmp_path, executed, error):
    source, notebook = _sample(tmp_path, executed=executed, error=error)
    figure = ROOT / "example_generated_notebooks" / "assets" / "synthetic_training_curves.png"
    with pytest.raises(ValueError, match="not completely executed|execution error"):
        renderer.render(source, notebook, figure)


def test_renderer_refuses_mismatched_hy_source(tmp_path):
    source, notebook = _sample(tmp_path)
    source.write_text('(print "modified after notebook was executed")\n')
    figure = ROOT / "example_generated_notebooks" / "assets" / "synthetic_training_curves.png"
    with pytest.raises(ValueError, match="provenance"):
        renderer.render(source, notebook, figure)
