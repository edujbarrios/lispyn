"""Runtime Python dependency checks and Hy-to-Jupyter macro integration."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest

from lispyn import dependencies as deps
from lispyn.notebooks import build_notebook


def test_stdlib_math_is_available_without_pip(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("pip must not run for standard-library modules")
    monkeypatch.setattr(deps.subprocess, "run", fail)
    status = deps.ensure_package("math")
    assert "already installed in this environment" in status
    assert "math" in status
    math = deps.use_package("math", install=False)
    assert math.sqrt(81) == 9.0
    assert "Requirement already installed" in capsys.readouterr().out


def test_installed_version_constraint_and_import_alias(capsys):
    status = deps.ensure_package("packaging>=23", import_name="packaging", install=False)
    assert "already installed" in status
    status = deps.ensure_package("pip", install=False)
    assert "already installed" in status
    math = deps.use_package("math", import_name="math", install=False)
    assert math.factorial(5) == 120


def test_absent_package_with_install_disabled(monkeypatch):
    monkeypatch.setattr(deps, "_available", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="Missing requirement"):
        deps.ensure_package("lispyn-never-install-package", install=False)


def test_pip_runs_in_current_interpreter_only_when_missing(monkeypatch):
    calls = []
    count = 0

    def available(*args):
        nonlocal count
        count += 1
        return ((True, "1.5") if count > 1 else (False, None))

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(deps, "_available", available)
    monkeypatch.setattr(deps.subprocess, "run", fake_run)

    status = deps.ensure_package("example-pkg>=1.2", import_name="example_pkg")
    assert "Installed requirement" in status
    assert len(calls) == 1
    assert calls[0][0][:4] == [sys.executable, "-m", "pip", "install"]
    assert calls[0][0][-1] == "example-pkg>=1.2"
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False


def test_pip_failure_has_actionable_error(monkeypatch):
    monkeypatch.setattr(deps, "_available", lambda *args: (False, None))
    monkeypatch.setattr(deps.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stderr="No matching distribution found", stdout=""
    ))
    with pytest.raises(RuntimeError, match="Failed to install"):
        deps.ensure_package("example-pkg")


@pytest.mark.parametrize("bad", [
    "", "-r requirements.txt", "foo @ https://example.com/a.whl",
    'numpy; python_version >= "3.10"',
])
def test_invalid_requirements_are_rejected(bad):
    with pytest.raises(ValueError):
        deps.ensure_package(bad)


def test_module_name_validation():
    with pytest.raises(ValueError, match="import_name"):
        deps.ensure_package("numpy", import_name="; rm -rf")
    with pytest.raises(ValueError, match="import_name"):
        deps.ensure_package("numpy", import_name="a/b")


def test_macro_compilation_and_real_hy_import(tmp_path):
    if not shutil.which("hy"):
        pytest.skip("Hy executable missing")
    source = tmp_path / "dependency.hy"
    source.write_text(
        '(require lispyn.macros [md ensure-package use-package])\n'
        '(md "# Dependencies" "A **real Markdown** cell.")\n'
        '(require lispyn.macros [ensure-package use-package])\n'
        '(setv math (use-package "math" :install False))\n'
        '(assert (= (math.sqrt 16) 4.0))\n'
        '(ensure-package "packaging>=23" :install False)\n'
        '(print "dependency-macro-ok")\n',
        encoding="utf-8",
    )
    result = subprocess.run(["hy", str(source)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Requirement already installed" in result.stdout
    assert "dependency-macro-ok" in result.stdout

    nb = build_notebook(source)
    nbformat.validate(nb)
    assert any(cell.cell_type == "markdown" and
               "Dependencies" in cell.source for cell in nb.cells)
    cell_src = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "ensure_package" in cell_src
    assert "use_package" in cell_src
    assert all(cell.get("execution_count") is None for cell in nb.cells
               if cell.cell_type == "code")


def test_generated_resnet_has_dependency_cell_and_no_executed_training():
    source = Path(__file__).resolve().parents[1] / "examples" / "resnet_finetune_300.hy"
    nb = build_notebook(source)
    cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    translated = "\n".join(cell.source for cell in cells)
    assert "ensure_package" in translated
    assert "use_package" in translated
    assert "300" in translated
    assert all(not cell.get("outputs") for cell in cells)
