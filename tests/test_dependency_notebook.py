"""Dependency macros compile to Python and run safely inside Jupyter."""

from pathlib import Path
import subprocess
import shutil

import nbformat


def test_requirements_in_executed_jupyter_notebook(tmp_path):
    source = tmp_path / "dependencies.hy"
    source.write_text(
        '(require lispyn.macros [md])\n'
        '(md "# Setup" "Runtime checks, no actual pip download.")\n'
        '(require lispyn.macros [ensure-package use-package])\n'
        '(setv math (use-package "math" :install False))\n'
        '(ensure-package "packaging>=23" :install False)\n'
        '(print "sqrt:" (math.sqrt 49))\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [shutil.which("lispyn"), "execute", str(source), "--timeout", "30"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    notebook = nbformat.read(source.with_suffix(".ipynb"), as_version=4)
    assert any(c.cell_type == "markdown" and "Setup" in c.source for c in notebook.cells)
    outputs = "\n".join(str(output.get("text", ""))
                        for cell in notebook.cells if cell.cell_type == "code"
                        for output in cell.get("outputs", []))
    assert "Requirement already installed in this environment: math" in outputs
    assert "Requirement already installed in this environment: packaging" in outputs
    assert "sqrt: 7.0" in outputs


def test_readme_links_runnable_quickstart_instead_of_embedding_large_resnet_recipe():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    example = root / "examples" / "quickstart.hy"
    catalog = root / "supported_macros" / "README.md"

    assert example.is_file() and "(train-until " in example.read_text()
    assert catalog.is_file()
    assert "examples/quickstart.hy" in readme
    assert "supported_macros/README.md" in readme
    assert "lispyn execute examples/quickstart.hy" in readme
    # Full multi-hundred-line ResNet recipes belong in examples/, not the landing page.
    assert (root / "examples" / "resnet_finetune_300.hy").is_file()
