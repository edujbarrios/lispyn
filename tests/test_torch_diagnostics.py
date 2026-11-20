"""CPU PyTorch diagnostics and safe inference, with Hy/Jupyter smoke tests."""

from pathlib import Path
import shutil
import subprocess

import pytest

from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "torch_diagnostics.hy"


def test_hy_diagnostics_compile_to_unexecuted_python_notebook():
    notebook = build_notebook(EXAMPLE)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert sum(c.cell_type == "markdown" for c in notebook.cells) >= 3
    for name in (
        "gradient_report", "assert_gradients", "model_footprint",
        "predict_batches", "temporarily_freeze", "deterministic_algorithms",
        "detect_anomaly",
    ):
        assert name in code, name
    assert all(c.execution_count is None and not c.outputs
               for c in notebook.cells if c.cell_type == "code")


def test_gradient_inspection_and_fail_closed_checks():
    torch = pytest.importorskip("torch")
    from lispyn.torch_diagnostics import assert_gradients, gradient_report

    model = torch.nn.Linear(2, 1)
    initial = gradient_report(model)
    assert initial["parameters_with_grad"] == 0
    assert initial["global_l2_norm"] is None
    with pytest.raises(ValueError, match="No gradients"):
        assert_gradients(model)
    assert_gradients(model, require_any=False) == initial
    with pytest.raises(TypeError, match="bool"):
        assert_gradients(model, require_any="yes")

    model(torch.ones(3, 2)).sum().backward()
    before = [p.grad.clone() for p in model.parameters()]
    report = gradient_report(model)
    assert report["parameters"] == 2
    assert report["parameters_with_grad"] == 2
    assert report["gradient_elements"] == 3
    assert report["nonfinite_elements"] == 0
    assert report["global_l2_norm"] > 0
    assert report["max_absolute"] > 0
    assert_gradients(model) == report
    assert all(torch.equal(a, p.grad) for a, p in zip(before, model.parameters()))

    with torch.no_grad():
        model.weight.grad[0, 0] = float("nan")
    nonfinite = gradient_report(model)
    assert nonfinite["nonfinite_elements"] == 1
    assert nonfinite["global_l2_norm"] is None
    assert not nonfinite["all_finite"]
    with pytest.raises(ValueError, match="Nonfinite gradients"):
        assert_gradients(model)


def test_model_footprint_includes_buffers_but_not_activation_memory():
    torch = pytest.importorskip("torch")
    from lispyn.torch_diagnostics import model_footprint

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 3, bias=False)  # 6 fp32 values
            self.register_buffer("cache", torch.ones(3, dtype=torch.float64))

        def forward(self, x):
            return self.layer(x)

    model = Model()
    info = model_footprint(model)
    assert info["parameter_elements"] == 6
    assert info["buffer_elements"] == 3
    assert info["parameter_bytes"] == 24
    assert info["buffer_bytes"] == 24
    assert info["total_bytes"] == 48
    assert info["total_mib"] > 0
    with pytest.raises(TypeError, match="torch.nn.Module"):
        model_footprint(object())


def test_scoped_freeze_restores_individual_flags_after_exception():
    torch = pytest.importorskip("torch")
    from lispyn.torch_diagnostics import temporarily_freeze

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[0].weight.requires_grad_(False)
    initial = [p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError, match="deliberate"):
        with temporarily_freeze(model):
            assert all(not p.requires_grad for p in model.parameters())
            raise RuntimeError("deliberate")
    assert [p.requires_grad for p in model.parameters()] == initial


def test_deterministic_policy_restored_on_exception_and_nested_scope():
    torch = pytest.importorskip("torch")
    from lispyn.torch_diagnostics import deterministic_algorithms

    initial = (torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled())
    with pytest.raises(RuntimeError, match="deliberate"):
        with deterministic_algorithms(warn_only=True):
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.is_deterministic_algorithms_warn_only_enabled()
            with deterministic_algorithms():
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.is_deterministic_algorithms_warn_only_enabled()
            raise RuntimeError("deliberate")
    assert (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == initial
    with pytest.raises(TypeError, match="bool"):
        with deterministic_algorithms(warn_only=1):
            pass


def test_batched_prediction_cpu_modes_inputs_and_errors():
    torch = pytest.importorskip("torch")
    from lispyn.torch_diagnostics import predict_batches

    model = torch.nn.Linear(2, 2)
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.zeros(6, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model.train()
    out = predict_batches(model, loader, max_batches=2)
    assert tuple(out.shape) == (4, 2)
    assert out.device.type == "cpu"
    assert not out.requires_grad
    assert model.training
    torch.testing.assert_close(out, model(x[:4]))
    model.eval()
    with pytest.raises(ValueError, match="input_key"):
        predict_batches(model, [{"x": x[:2]}])
    assert not model.training
    out2 = predict_batches(model, [{"x": x[:2]}], input_key="x")
    assert tuple(out2.shape) == (2, 2)
    assert not model.training
    with pytest.raises(KeyError, match="missing"):
        predict_batches(model, [{"x": x[:2]}], input_key="missing")
    with pytest.raises(ValueError, match="empty"):
        predict_batches(model, [])
    with pytest.raises(ValueError, match="max_batches"):
        predict_batches(model, [x], max_batches=0)
    with pytest.raises(ValueError, match="max_batches"):
        predict_batches(model, [x], max_batches=True)
    assert not model.training

    # Inference must also preserve intentionally mixed child-module modes.
    mixed = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
    mixed.train()
    mixed[0].eval()
    predict_batches(mixed, [x[:2]])
    assert mixed.training
    assert not mixed[0].training
    assert mixed[1].training


def test_example_executes_in_direct_hy_on_cpu(tmp_path):
    pytest.importorskip("torch")
    hy = shutil.which("hy")
    assert hy
    copy = tmp_path / EXAMPLE.name
    copy.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([hy, str(copy)], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Batched inference:" in result.stdout
    assert "Trainable flags and policy restored" in result.stdout
