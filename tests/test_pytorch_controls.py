"""PyTorch control macro compilation and CPU behavior."""

from pathlib import Path
import shutil
import subprocess

import pytest

from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]


def test_new_macros_compile_into_native_python_notebook(tmp_path):
    source = ROOT / "examples" / "pytorch_controls.hy"
    notebook = build_notebook(source)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for expression in (
        "torch.autocast", "torch.no_grad", "clip_gradients",
        "set_module_trainable", "parameter_counts", "move_to_device",
        "save_checkpoint", "load_checkpoint",
    ):
        assert expression in code, expression
    assert all(cell.get("execution_count") is None
               for cell in notebook.cells if cell.cell_type == "code")


def test_native_torch_helpers_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.torch_utils import (
        clip_gradients, move_to_device, parameter_counts, set_module_trainable
    )
    from lispyn.training import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 1)
    assert parameter_counts(model) == {"total": 3, "trainable": 3, "frozen": 0}
    assert set_module_trainable(model, False) is model
    assert parameter_counts(model) == {"total": 3, "trainable": 0, "frozen": 3}
    assert model.training, "freezing should not silently enable eval()"
    assert set_module_trainable(model, True) is model
    with pytest.raises(TypeError, match="torch.nn.Module"):
        parameter_counts(object())
    with pytest.raises(TypeError, match="bool"):
        set_module_trainable(model, 1)

    data = {"inputs": [torch.tensor([1.0]), ("label", torch.tensor([2.0]))]}
    moved = move_to_device(data, "cpu")
    assert moved is not data
    assert moved["inputs"][1][0] == "label"
    assert moved["inputs"][0].device.type == "cpu"
    assert moved["inputs"][0] is not data["inputs"][0] or moved["inputs"][0].device.type == "cpu"
    with pytest.raises(TypeError, match="bool"):
        move_to_device(data, "cpu", non_blocking="yes")

    with pytest.raises(ValueError, match="No gradients"):
        clip_gradients(model, 1.0)
    with pytest.raises(ValueError, match="max_norm"):
        clip_gradients(model, float("nan"))
    with pytest.raises(ValueError, match="max_norm"):
        clip_gradients(model, True)
    with pytest.raises(ValueError, match="norm_type"):
        clip_gradients(model, 1.0, norm_type=0)
    model(torch.ones(1, 2)).sum().backward()
    norm = clip_gradients(model, 0.1)
    assert norm > 0
    assert torch.linalg.vector_norm(torch.cat([
        p.grad.flatten() for p in model.parameters()
    ])).item() <= 0.10001

    path = tmp_path / "model.pt"
    save_checkpoint(model, path, history={"loss": [0.5]})
    with torch.no_grad():
        model.bias.add_(100)
    history = load_checkpoint(model, path)
    assert history == {"loss": [0.5]}


def test_hy_cpu_controls_run_and_restore_grad_state(tmp_path):
    pytest.importorskip("torch")
    hy = shutil.which("hy")
    assert hy
    source = tmp_path / "guards.hy"
    source.write_text(
        '(require lispyn.macros [with-no-grad with-autocast])\n'
        '(import torch)\n'
        '(assert (torch.is-grad-enabled))\n'
        '(try\n'
        '  (with-no-grad\n'
        '    (assert (not (torch.is-grad-enabled)))\n'
        '    (raise (RuntimeError "expected")))\n'
        '  (except [RuntimeError] None))\n'
        '(assert (torch.is-grad-enabled))\n'
        '(setv x (torch.ones [2 2]))\n'
        '(setv y (with-autocast "cpu" torch.bfloat16 (torch.mm x x)))\n'
        '(assert (= y.dtype torch.bfloat16))\n'
        '(print "contexts-ok")\n',
        encoding="utf-8",
    )
    r = subprocess.run([hy, str(source)], cwd=tmp_path,
                       text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "contexts-ok" in r.stdout


def test_complete_cpu_example_runs(tmp_path):
    pytest.importorskip("torch")
    hy = shutil.which("hy")
    assert hy
    source = ROOT / "examples" / "pytorch_controls.hy"
    # Run from a scratch copy: checkpoints must not modify the repository.
    copy = tmp_path / source.name
    copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([hy, str(copy)], cwd=tmp_path,
                       text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "PyTorch controls example passed" in r.stdout
    assert (tmp_path / "artifacts" / "controls.pt").is_file()
