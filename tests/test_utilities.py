"""Tests for dependency-free Hy diagnostics and the beginner workflow."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lispyn.utilities import assert_close, timing

ROOT = Path(__file__).resolve().parents[1]
HELLO = ROOT / "examples" / "hello.hy"


def test_assert_close_scalar_validation():
    assert assert_close(0.1 + 0.2, 0.3)
    assert assert_close(10, 10)
    assert assert_close(10.1, 10, abs_tol=0.2)
    with pytest.raises(AssertionError, match="actual=10"):
        assert_close(10, 11)
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            assert_close(invalid, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        assert_close(1, 1, rel_tol=-1)
    with pytest.raises(TypeError, match="bool"):
        assert_close(True, 1)


def test_timing_prints_even_when_body_raises(capsys):
    with timing("example"):
        pass
    assert "example:" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="boom"):
        with timing("failed example"):
            raise RuntimeError("boom")
    assert "failed example:" in capsys.readouterr().out
    with pytest.raises(ValueError, match="nonempty"):
        with timing("  "):
            pass


def test_hello_hy_direct_and_notebook_build(tmp_path):
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook
    notebook = build_notebook(HELLO)
    assert notebook.cells[0].cell_type == "markdown"
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "assert_close" in code
    assert "timing" in code
    assert "Hello from Lispyn!" in code

    result = subprocess.run(
        [sys.executable, "-m", "lispyn.cli", "run", str(HELLO)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "Hello from Lispyn!" in result.stdout
    assert "Lispyn checks passed!" in result.stdout

    output = tmp_path / "hello.ipynb"
    result = subprocess.run(
        [sys.executable, "-m", "lispyn.cli", "build", str(HELLO), "-o", str(output)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()


def test_hello_hy_notebook_execute(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    if shutil.which("jupyter") is None:
        pytest.skip("Jupyter unavailable")
    output = tmp_path / "hello.ipynb"
    result = subprocess.run(
        [sys.executable, "-m", "lispyn.cli", "execute", str(HELLO),
         "-o", str(output), "--timeout", "90"],
        capture_output=True, text=True, cwd=ROOT, timeout=150,
    )
    assert result.returncode == 0, result.stderr
    import nbformat
    notebook = nbformat.read(output, as_version=4)
    outputs = "\n".join(str(out.get("text", "")) for c in notebook.cells
                        if c.cell_type == "code" for out in c.get("outputs", []))
    assert "Lispyn checks passed!" in outputs
