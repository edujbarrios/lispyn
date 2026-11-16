"""README quickstart outputs must come from verified, executed Jupyter cells."""
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


def _sample(tmp_path, *, executed=True, error=False, image=True, scores=True):
    source = tmp_path / "study.hy"
    source.write_text(
        '(require lispyn.macros [train-until])\n'
        '(setv run (train-until model training 3 :epochs 10))\n'
        '(require lispyn.macros [md])\n'
        '(md "Done")\n'
        '(require lispyn.macros [curve])\n'
        '(curve run :metric "loss")\n',
        encoding="utf-8",
    )
    outputs = [
        new_output(
            "stream", name="stdout",
            text="Samples: 8 train, 4 validation, 4 held-out test\n"
                 "Completed epochs: 10\n",
        ),
        new_output(
            "display_data",
            data={"text/markdown":
                  "# PyTorch baseline · recorded metrics\n\n"
                  "**Epochs recorded:** 10\n"},
        ),
        new_output(
            "display_data",
            data={"text/markdown": "Held-out test accuracy: **50.0%**"},
        ),
    ]
    if scores:
        outputs.append(new_output(
            "display_data",
            data={"text/markdown":
                  "### Recorded experiment scores\n\n"
                  "| Metric | Value |\n| --- | ---: |\n"
                  "| test_accuracy | 0.5 |\n"},
        ))
    if image:
        outputs.append(new_output(
            "display_data",
            data={"image/png": "aGVsbG8="},
        ))
    if error:
        outputs.append(new_output(
            "error", ename="RuntimeError", evalue="oops",
            traceback=["RuntimeError: oops"],
        ))
    code = new_code_cell(
        "from lispyn.plots import plot_history\nplot_history(run, metric='loss')",
        execution_count=1 if executed else None,
        outputs=outputs if executed else [],
    )
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


def _figure():
    return ROOT / "example_generated_notebooks" / "assets" / "synthetic_training_curves.png"


def test_renderer_shows_short_verified_hy_snippet_and_one_real_chart(tmp_path):
    source, notebook = _sample(tmp_path)
    section = renderer.render(source, notebook, _figure())
    assert "**Hy syntax**" in section
    assert "(train-until model training 3" in section
    assert "(curve run" in section
    assert section.index("(train-until") < section.index("Samples: 8 train")
    assert "Completed epochs: 10" in section
    assert "Held-out test accuracy: **50.0%**" in section
    assert section.count("![Actual training and validation loss") == 1
    assert "quickstart_executed.ipynb" in section
    assert "PyTorch baseline · recorded metrics" not in section
    assert "test_accuracy | 0.5" not in section
    assert "Cell 1 · Markdown" not in section
    assert "<details>" not in section
    assert "aGVsbG8" not in section

    readme = tmp_path / "README.md"
    readme.write_text(
        f"Intro\n{renderer.BEGIN}\nplaceholder\n{renderer.END}\nEnd\n"
    )
    renderer.rewrite(readme, section)
    renderer.rewrite(readme, section, check=True)
    assert section in readme.read_text()
    assert readme.read_text().count("Held-out test accuracy") == 1
    with pytest.raises(ValueError, match="stale"):
        renderer.rewrite(readme, section + "unverified values", check=True)


@pytest.mark.parametrize("executed,error", [
    (False, False),
    (True, True),
])
def test_renderer_refuses_missing_execution_or_errors(tmp_path, executed, error):
    source, notebook = _sample(tmp_path, executed=executed, error=error)
    with pytest.raises(ValueError, match="executed|execution error"):
        renderer.render(source, notebook, _figure())


@pytest.mark.parametrize("image,scores,match", [
    (False, True, "recorded training plot"),
    (True, False, "metric table"),
])
def test_renderer_refuses_missing_figure_or_real_metrics(tmp_path, image, scores, match):
    source, notebook = _sample(tmp_path, image=image, scores=scores)
    with pytest.raises(ValueError, match=match):
        renderer.render(source, notebook, _figure())


def test_renderer_refuses_mismatched_hy_source(tmp_path):
    source, notebook = _sample(tmp_path)
    source.write_text('(print "modified after notebook was executed")\n')
    with pytest.raises(ValueError, match="provenance"):
        renderer.render(source, notebook, _figure())


def test_committed_readme_matches_genuine_executed_quickstart():
    """Fail CI if the public README drifts away from its real notebook."""
    renderer.rewrite(renderer.README, renderer.render(), check=True)
