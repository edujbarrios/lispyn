"""Core tests deliberately avoid GPU, network, and long model training."""

from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest

from lispyn import cli
from lispyn.compiler import LispynError, check_source
from lispyn.notebooks import build_notebook, execute_notebook, write_notebook


def test_missing_and_non_hy_input(tmp_path):
    with pytest.raises(LispynError):
        check_source(tmp_path / "missing.hy")
    (tmp_path / "file.py").write_text("pass")
    with pytest.raises(LispynError):
        check_source(tmp_path / "file.py")


def test_build_creates_python_notebook_with_attribution(tmp_path):
    source = tmp_path / "trial.hy"
    source.write_text('(print "hello")')
    with patch("lispyn.notebooks.compile_hy", return_value='print("hello")\n'):
        nb = build_notebook(source)
    assert nb.cells[0].cell_type == "markdown"
    assert "Eduardo J. Barrios" in nb.cells[0].source
    assert nb.cells[1].source == 'print("hello")\n'
    path = write_notebook(nb, tmp_path / "test.ipynb")
    saved = nbformat.read(path, as_version=4)
    nbformat.validate(saved)
    assert saved.metadata.lispyn.source == "trial.hy"


def test_build_does_not_run_code(tmp_path):
    source = tmp_path / "trial.hy"
    source.write_text('(print "hello")')
    with patch("lispyn.notebooks.compile_hy", return_value='raise RuntimeError("must not run")'):
        assert cli.main(["build", str(source)]) == 0
    output = nbformat.read(tmp_path / "trial.ipynb", as_version=4)
    assert output.cells[1].execution_count is None
    assert output.cells[1].outputs == []


def test_execute_runs_python_cells_in_real_kernel(tmp_path):
    source = tmp_path / "trial.hy"
    source.write_text('(print "hello")')
    with patch("lispyn.notebooks.compile_hy", return_value='print("hello from Jupyter")'):
        nb = build_notebook(source)
    try:
        execute_notebook(nb, cwd=tmp_path, timeout=30)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Jupyter kernel not available: {exc}")
    assert any("hello from Jupyter" in str(o.get("text", "")) for o in nb.cells[1].outputs)


def test_compile_hy_invoked_without_running_program(tmp_path):
    from lispyn.compiler import compile_hy

    source = tmp_path / "experiment.hy"
    source.write_text("(print 1)")
    with patch("lispyn.compiler.shutil.which", return_value="/usr/bin/hy2py"), patch(
        "lispyn.compiler.subprocess.run"
    ) as invoke:
        invoke.return_value.returncode = 0
        invoke.return_value.stdout = "print(1)\n"
        assert compile_hy(source) == "print(1)\n"
        assert invoke.call_args.args[0] == ["/usr/bin/hy2py", str(source.resolve())]


def test_data_csv_and_tensor(tmp_path):
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    from lispyn.data import read_dataset, to_tensor

    filename = tmp_path / "tiny.csv"
    filename.write_text("x,y\n1,2\n3,4\n")
    frame = read_dataset(filename)
    assert frame.shape == (2, 2)
    values = frame.to_numpy(dtype="float32")
    tensor = to_tensor(values)
    assert tuple(tensor.shape) == (2, 2)
    assert tensor.dtype == torch.float32
