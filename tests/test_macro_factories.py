"""Generated model/optimizer/plot macros, direct Hy and Jupyter integration."""
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
import pytest

from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase.hy"


def _hy(source, cwd):
    executable = shutil.which("hy")
    assert executable, "Hy must be installed for the macro test suite"
    result = subprocess.run(
        [executable, str(source)], cwd=cwd,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_showcase_compiles_to_native_python_with_dependency_cell():
    notebook = build_notebook(SHOWCASE)
    nbformat.validate(notebook)
    assert sum(cell.cell_type == "markdown" for cell in notebook.cells) >= 2
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for fragment in ("def make_mlp", "torch.nn.Sequential",
                     "def make_adam", "torch.optim.Adam",
                     "def loss_chart", ".subplots()", ".tight_layout()",
                     ".savefig(", "def train_one"):
        assert fragment in code, fragment
    assert "lispyn.factory" not in code
    assert "lispyn.plots" not in code
    assert any("lispyn-pip-install" in c.metadata.get("tags", [])
               for c in notebook.cells if c.cell_type == "code")
    assert all(c.execution_count is None and not c.outputs
               for c in notebook.cells if c.cell_type == "code")


def test_matplotlib_scopes_and_plot_factories_without_torch(tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "figures.hy"
    source.write_text(
        '(require lispyn.macros [with-figure defplot])\n'
        '(import matplotlib.pyplot :as plt)\n'
        '(defplot make-chart [points] [figure axes]\n'
        '  (.plot axes [0 1 2] points)\n'
        '  (.set_title axes "Series"))\n'
        '(setv first (make-chart [1 2 3]))\n'
        '(setv second (make-chart [3 2 1]))\n'
        '(assert (not (is first second)))\n'
        '(plt.close first)\n'
        '(plt.close second)\n'
        '(setv original (set (plt.get_fignums)))\n'
        '(setv seen [])\n'
        '(defn evaluate-once [] (.append seen 1) [1 2 3])\n'
        '(setv figure (with-figure [f ax]\n'
        '  (.plot ax [0 1 2] (evaluate-once))))\n'
        '(assert (= seen [1]))\n'
        '(plt.close figure)\n'
        '(try\n'
        '  (with-figure [f ax] (raise (RuntimeError "plot failed")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (= original (set (plt.get_fignums))))\n'
        '(defplot broken [] [f ax] (raise (RuntimeError "bad")))\n'
        '(try (broken) (except [RuntimeError] None))\n'
        '(assert (= original (set (plt.get_fignums))))\n'
        '(print "native-plot-macros-passed")\n',
        encoding="utf-8",
    )
    assert "native-plot-macros-passed" in _hy(source, tmp_path)


def test_invalid_macro_signatures_fail_at_compile_time(tmp_path):
    cases = [
        '(defmodel make-network [x])',
        '(defoptimizer make-optimizer [model] torch.optim.Adam)',
        '(with-figure [fig] (print 1))',
        '(defplot make-chart [] [fig] (print 1))',
    ]
    for index, form in enumerate(cases):
        source = tmp_path / f"invalid{index}.hy"
        source.write_text('(require lispyn.macros [defmodel defoptimizer with-figure defplot])\n'
                          + form + '\n', encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "lispyn.cli", "build", str(source)],
            cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0, form


def test_pytorch_factories_train_with_independent_models(tmp_path):
    torch = pytest.importorskip("torch")
    assert torch is not None
    source = tmp_path / "models.hy"
    source.write_text(
        '(require lispyn.macros [defmodel defoptimizer deftrain-step collect-epochs])\n'
        '(import torch)\n'
        '(defmodel make-network [inputs hidden]\n'
        '  (torch.nn.Linear inputs hidden)\n'
        '  (torch.nn.ReLU)\n'
        '  (torch.nn.Linear hidden 1))\n'
        '(setv one (make-network 2 4))\n'
        '(setv two (make-network 2 4))\n'
        '(assert (not (is (get one 0) (get two 0))))\n'
        '(defoptimizer opt [m rate] torch.optim.SGD :momentum 0.9)\n'
        '(setv optimizer (opt one 0.01))\n'
        '(assert (= (get optimizer.defaults "momentum") 0.9))\n'
        '(assert (= (get optimizer.defaults "lr") 0.01))\n'
        '(setv x (torch.ones [3 2]))\n'
        '(setv y (torch.ones [3 1]))\n'
        '(setv criterion (torch.nn.MSELoss))\n'
        '(deftrain-step train-one one optimizer criterion)\n'
        '(setv history (collect-epochs epoch 3 (train-one x y)))\n'
        '(assert (= (len history) 3))\n'
        '(print "model-factories-passed")\n',
        encoding="utf-8",
    )
    assert "model-factories-passed" in _hy(source, tmp_path)


def test_showcase_runs_and_executes_to_notebook_with_real_pngs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("nbclient")
    source = tmp_path / "showcase.hy"
    source.write_text(SHOWCASE.read_text(encoding="utf-8"), encoding="utf-8")
    output = _hy(source, tmp_path)
    assert "Training losses:" in output
    assert "Showcase generated two PNG charts" in output
    for name in ("showcase-loss.png", "showcase-comparison.png"):
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size > 1000

    notebook_path = tmp_path / "showcase-executed.ipynb"
    cli = [sys.executable, "-m", "lispyn.cli"]
    run = subprocess.run(
        cli + ["execute", str(source), "-o", str(notebook_path), "--timeout", "90"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    assert run.returncode == 0, run.stderr
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert any("Showcase generated two PNG charts" in str(o.get("text", ""))
               for cell in notebook.cells if cell.cell_type == "code"
               for o in cell.get("outputs", []))
