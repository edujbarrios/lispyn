"""Macro-first Hy code generation and scope hygiene, with CPU integration tests."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "macro_first_training.hy"


def _run_hy(source: Path, cwd: Path):
    executable = shutil.which("hy")
    assert executable, "Expected Hy executable in the installed test environment"
    result = subprocess.run([executable, str(source)], cwd=cwd,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_macro_first_example_compiles_to_self_contained_python():
    notebook = build_notebook(EXAMPLE)
    assert sum(c.cell_type == "markdown" for c in notebook.cells) >= 3
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    for expected in (
        "def train_one", ".backward()", ".zero_grad()", ".step()",
        "torch.inference_mode", "torch.enable_grad",
        "torch.random.fork_rng", "torch.use_deterministic_algorithms",
        ".requires_grad_", "range(4)", "assert",
    ):
        assert expected in code, expected
    assert "lispyn.torch_diagnostics" not in code, (
        "Macro-first execution scopes must compile to native Python")
    assert all(c.execution_count is None and not c.outputs
               for c in notebook.cells if c.cell_type == "code")


def test_example_runs_cpu_without_downloads(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / EXAMPLE.name
    source.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    output = _run_hy(source, tmp_path)
    assert "Macro-generated training loss:" in output
    assert "Custom Hy macro and generated PyTorch code passed" in output


def test_macros_restore_state_on_exceptions_and_evaluate_once(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "macro_hygiene.hy"
    source.write_text(
        '(require lispyn.macros [with-eval with-training with-grad\n'
        '  with-rng-fork with-parameter-freeze with-determinism\n'
        '  optimizer-step collect-epochs])\n'
        '(import torch)\n'
        '(setv model (torch.nn.Sequential (torch.nn.Dropout) (torch.nn.Linear 2 1)))\n'
        '(.eval (get model 0))\n'
        '(setv flags (list (map (fn [m] m.training) (.modules model))))\n'
        '(setv p (next (iter (.parameters model))))\n'
        '(setv initial-grad p.requires_grad)\n'
        '(try\n'
        '  (with-eval model\n'
        '    (raise (RuntimeError "eval-error")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (= (list (map (fn [m] m.training) (.modules model))) flags))\n'
        '(try\n'
        '  (with-training model\n'
        '    (assert (getattr (get model 0) \"training\"))\n'
        '    (raise (RuntimeError "train-error")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (= (list (map (fn [m] m.training) (.modules model))) flags))\n'
        '(try\n'
        '  (with-parameter-freeze model\n'
        '    (assert (not p.requires_grad))\n'
        '    (raise (RuntimeError "freeze-error")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (= p.requires_grad initial-grad))\n'
        '(setv enabled (torch.are_deterministic_algorithms_enabled))\n'
        '(setv warning (torch.is_deterministic_algorithms_warn_only_enabled))\n'
        '(try\n'
        '  (with-determinism\n'
        '    (assert (torch.are_deterministic_algorithms_enabled))\n'
        '    (raise (RuntimeError "determinism-error")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (= enabled (torch.are_deterministic_algorithms_enabled)))\n'
        '(assert (= warning (torch.is_deterministic_algorithms_warn_only_enabled)))\n'
        '(setv rand-before (torch.get_rng_state))\n'
        '(try\n'
        '  (with-rng-fork\n'
        '    (torch.manual_seed 567)\n'
        '    (raise (RuntimeError "rng-error")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (torch.equal rand-before (torch.get_rng_state)))\n'
        '(with [(torch.no_grad)]\n'
        '  (with-grad (assert (torch.is-grad_enabled)))\n'
        '  (assert (not (torch.is_grad_enabled))))\n'
        '(setv source [(torch.tensor [[1.0 2.0]]) (torch.tensor [[1.0]])])\n'
        '(setv optimizer (torch.optim.SGD (.parameters model) :lr 0.01))\n'
        '(setv criterion (torch.nn.MSELoss))\n'
        '(setv visits [])\n'
        '(defn get-opt [] (.append visits "optimizer") optimizer)\n'
        '(defn get-loss []\n'
        '  (.append visits "loss")\n'
        '  (criterion (model (get source 0)) (get source 1)))\n'
        '(setv detached-loss (optimizer-step (get-opt) (get-loss)))\n'
        '(assert (= visits ["optimizer" "loss"]))\n'
        '(assert (not detached-loss.requires_grad))\n'
        '(setv lispyn-results "original")\n'
        '(setv epochs (collect-epochs epoch 3 (* epoch 2)))\n'
        '(assert (= epochs [0 2 4]))\n'
        '(assert (= lispyn-results "original"))\n'
        '(print "macro-hygiene-passed")\n',
        encoding="utf-8",
    )
    assert "macro-hygiene-passed" in _run_hy(source, tmp_path)


def test_no_new_backend_module_is_needed_for_new_macros():
    macro_source = (ROOT / "src/lispyn/macros.hy").read_text(encoding="utf-8")
    assert len(re.findall(r"(?m)^\(defmacro ", macro_source)) >= 78
    for macro in (
        "with-training", "with-grad", "with-rng-fork", "optimizer-step",
        "deftrain-step", "collect-epochs",
    ):
        assert f"(defmacro {macro} [" in macro_source
