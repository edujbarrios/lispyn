"""Standalone pip-install cells, including no-download Jupyter execution."""

from pathlib import Path
import shutil
import subprocess

import nbformat
import pytest

from lispyn import dependencies
from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook, split_markdown_macros


def _source():
    return (
        '(require lispyn.macros [md pip-install])\n'
        '(md "## Set up dependencies" "Use the active notebook kernel.")\n'
        '(pip-install "math" "packaging>=23")\n'
        '(require lispyn.macros [use-package])\n'
        '(setv math (use-package "math" :install False))\n'
        '(print "sqrt:" (math.sqrt 49))\n'
    )


def test_pip_directive_makes_its_own_python_code_cell(tmp_path):
    src = tmp_path / "deps.hy"
    src.write_text(_source(), encoding="utf-8")
    nb = build_notebook(src)
    nbformat.validate(nb)

    deps_cells = [cell for cell in nb.cells
                  if cell.cell_type == "code" and
                  "lispyn-pip-install" in cell.metadata.get("tags", [])]
    assert len(deps_cells) == 1
    cell = deps_cells[0]
    assert "from lispyn.dependencies import ensure_packages" in cell.source
    assert "ensure_packages('math', 'packaging>=23')" in cell.source
    assert nb.cells.index(cell) < next(
        i for i, c in enumerate(nb.cells) if
        c.cell_type == "code" and "sqrt:" in c.source
    )
    assert all(c.get("execution_count") is None
               for c in nb.cells if c.cell_type == "code")
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_pip_macro_executes_directly_without_downloading(tmp_path):
    hy = shutil.which("hy")
    assert hy is not None
    src = tmp_path / "deps.hy"
    src.write_text(_source(), encoding="utf-8")
    result = subprocess.run([hy, str(src)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Requirement already installed in this environment: math" in result.stdout
    assert "Requirement already installed in this environment: packaging" in result.stdout
    assert "sqrt: 7.0" in result.stdout


def test_pip_macro_executes_inside_actual_jupyter(tmp_path):
    cli = shutil.which("lispyn")
    assert cli is not None
    src = tmp_path / "deps.hy"
    src.write_text(_source(), encoding="utf-8")
    result = subprocess.run(
        [cli, "execute", str(src), "--timeout", "45"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    nb = nbformat.read(src.with_suffix(".ipynb"), as_version=4)
    pip_cell = next(c for c in nb.cells if
                    "lispyn-pip-install" in c.metadata.get("tags", []))
    assert pip_cell.execution_count is not None
    stdout = "".join(str(o.get("text", "")) for c in nb.cells if c.cell_type == "code"
                     for o in c.get("outputs", []))
    assert "already installed in this environment: math" in stdout
    assert "sqrt: 7.0" in stdout


@pytest.mark.parametrize("source, message", [
    ('(pip-install)\n', "literal package"),
    ('(pip-install (+ 1 2))\n', "literal package"),
    ('(pip-install "-r requirements.txt")\n', "Invalid pip-install cell"),
    ('(pip-install "foo @ https://example.com/a.whl")\n', "Invalid pip-install cell"),
    ('(do\n  (pip-install "math"))\n', "top-level"),
    ('(pip-install "math") (print 3)\n', "own lines"),
])
def test_pip_cell_rejects_unsafe_or_ambiguous_forms(source, message):
    with pytest.raises(LispynError, match=message):
        split_markdown_macros(source)


def test_multi_requirements_validate_before_any_install(monkeypatch):
    installed = []
    monkeypatch.setattr(
        dependencies, "ensure_package",
        lambda requirement, **kwargs: installed.append(requirement),
    )
    with pytest.raises(ValueError):
        dependencies.ensure_packages("math", "-r requirements.txt")
    assert installed == []
    assert dependencies.ensure_packages("math", "packaging") == [None, None]
    assert installed == ["math", "packaging"]


def test_synthetic_plot_example_compiles_and_is_in_gallery():
    source = Path(__file__).resolve().parents[1] / "examples" / "synthetic_curves.hy"
    nb = build_notebook(source)
    assert any("lispyn-pip-install" in c.metadata.get("tags", [])
               for c in nb.cells if c.cell_type == "code")
    assert any("synthetic" in c.source.lower()
               for c in nb.cells if c.cell_type == "markdown")
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
