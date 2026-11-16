"""Validate guard behavior and actual Hy run/build/execute paths."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "validate_inputs.hy"


def test_finite_numeric_arrays_and_failures():
    np = pytest.importorskip("numpy")
    from lispyn.validation import assert_finite

    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert assert_finite(data) is data
    assert assert_finite([1, 2, 3]) == [1, 2, 3]
    assert assert_finite(np.array([1 + 1j, 2 + 0j])).shape == (2,)
    with pytest.raises(AssertionError, match="2 non-finite"):
        assert_finite(np.array([1, float("nan"), float("inf")]), name="sensors")
    with pytest.raises(ValueError, match="empty"):
        assert_finite(np.array([]))
    with pytest.raises(TypeError, match="numeric"):
        assert_finite(["private-string"])
    with pytest.raises(ValueError, match="name"):
        assert_finite(data, name="")


def test_shape_wildcards_and_invalid_dimensions():
    np = pytest.importorskip("numpy")
    from lispyn.validation import assert_shape

    arr = np.arange(6).reshape(2, 3)
    assert assert_shape(arr, [2, 3]) is arr
    assert assert_shape(arr, (None, 3)) is arr
    with pytest.raises(AssertionError, match="has shape"):
        assert_shape(arr, [3, 2], name="features")
    with pytest.raises(AssertionError, match="has shape"):
        assert_shape(arr, [2, 3, 1])
    for expected in ([True, 3], [-1, 3], ["2", 3], [], [None] * 9):
        with pytest.raises(ValueError, match="expected shape"):
            assert_shape(arr, expected)
    with pytest.raises(TypeError, match="shape attribute"):
        assert_shape([1, 2], [2])
    with pytest.raises(ValueError, match="name"):
        assert_shape(arr, [2, 3], name=" ")


def test_pandas_and_optional_torch_integration():
    pd = pytest.importorskip("pandas")
    from lispyn.validation import assert_shape
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert assert_shape(frame, [2, None]) is frame

    try:
        import torch
    except ImportError:
        return
    from lispyn.validation import assert_finite
    tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
    assert assert_finite(tensor) is tensor
    assert assert_shape(tensor, [None, 2]) is tensor
    assert tensor.requires_grad
    with pytest.raises(AssertionError, match="1 non-finite"):
        assert_finite(torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="empty"):
        assert_finite(torch.empty(0))


def test_hy_example_direct_build_and_executed_notebook(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("pandas")
    pytest.importorskip("nbclient")
    import nbformat
    from lispyn.notebooks import build_notebook

    notebook = build_notebook(EXAMPLE)
    nbformat.validate(notebook)
    assert any(c.cell_type == "markdown" for c in notebook.cells)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "lispyn.validation.assert_finite" in code
    assert "lispyn.validation.assert_shape" in code

    cli = [sys.executable, "-m", "lispyn.cli"]
    direct = subprocess.run(cli + ["run", str(EXAMPLE)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 0, direct.stderr
    assert "Research input checks passed." in direct.stdout

    template_path = tmp_path / "template.ipynb"
    build = subprocess.run(cli + ["build", str(EXAMPLE), "-o", str(template_path)],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stderr
    template = nbformat.read(template_path, as_version=4)
    assert all(c.execution_count is None and not c.outputs
               for c in template.cells if c.cell_type == "code")

    executed_path = tmp_path / "executed.ipynb"
    run = subprocess.run(cli + ["execute", str(EXAMPLE), "-o", str(executed_path),
                                "--timeout", "90"],
                         cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    executed = nbformat.read(executed_path, as_version=4)
    assert all(c.execution_count is not None for c in executed.cells
               if c.cell_type == "code")
    output = "\n".join(str(o.get("text", ""))
                       for c in executed.cells if c.cell_type == "code"
                       for o in c.get("outputs", []))
    assert "Research input checks passed." in output
