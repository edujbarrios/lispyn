"""Real PyTorch tests for gradient accumulation, clipping and LR schedules."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def _dataset():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0., 1.], [1., 0.], [1., 1.],
                      [0.2, 0.8], [0.8, 0.2], [0.9, 0.9]])
    y = torch.tensor([0, 1, 0, 0, 1, 1])
    return torch, torch.utils.data.TensorDataset(x, y)


def test_accumulation_matches_manual_two_optimizer_updates():
    torch, data = _dataset()
    from lispyn.training import fit

    torch.manual_seed(8)
    model = torch.nn.Linear(2, 2)
    expected = torch.nn.Linear(2, 2)
    expected.load_state_dict(model.state_dict())
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.12)
    manual_opt = torch.optim.SGD(expected.parameters(), lr=0.12)
    updates = []
    real_step = opt.step

    def step(*args, **kwargs):
        updates.append(True)
        return real_step(*args, **kwargs)

    opt.step = step
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    history = fit(model, loader, opt, loss_fn=criterion, epochs=1,
                  accumulation_steps=4)
    x, y = data.tensors
    for start, end in ((0, 4), (4, 6)):
        manual_opt.zero_grad()
        loss = criterion(expected(x[start:end]), y[start:end])
        loss.backward()
        manual_opt.step()
    assert len(updates) == 2
    assert len(history["train_loss"]) == 1
    for current, wanted in zip(model.parameters(), expected.parameters()):
        assert torch.allclose(current, wanted, atol=1e-6, rtol=1e-6)


def test_gradient_clip_bounds_an_optimizer_step():
    torch = pytest.importorskip("torch")
    from lispyn.training import fit

    model = torch.nn.Linear(1, 1)
    initial = [param.detach().clone() for param in model.parameters()]
    x = torch.tensor([[900.0], [1100.0]])
    y = torch.tensor([[0.0], [0.0]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    fit(model, loader, opt, loss_fn=torch.nn.MSELoss(), epochs=1,
        accumulation_steps=2, gradient_clip=0.25)
    displacement = torch.sqrt(sum(
        ((param.detach() - old) ** 2).sum()
        for param, old in zip(model.parameters(), initial)
    )).item()
    assert 0 < displacement <= 0.025001


@pytest.mark.parametrize("settings, message", [
    ({"accumulation_steps": 0}, "accumulation_steps"),
    ({"accumulation_steps": -1}, "accumulation_steps"),
    ({"accumulation_steps": True}, "accumulation_steps"),
    ({"accumulation_steps": 1.5}, "accumulation_steps"),
    ({"gradient_clip": 0}, "gradient_clip"),
    ({"gradient_clip": -2.0}, "gradient_clip"),
    ({"gradient_clip": float("nan")}, "gradient_clip"),
    ({"gradient_clip": float("inf")}, "gradient_clip"),
    ({"gradient_clip": True}, "gradient_clip"),
])
def test_invalid_accumulation_options_do_not_train(settings, message):
    from lispyn.training import fit
    with pytest.raises(ValueError, match=message):
        fit(None, [], None, **settings)


def test_cosine_schedule_checkpoint_history_and_manifest(tmp_path):
    torch, data = _dataset()
    from lispyn.workflow import train
    from lispyn.training import load_checkpoint

    result = train(
        torch.nn.Linear(2, 2), data,
        validation=data, loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=4, batch_size=1, lr=0.12, seed=17, device="cpu",
        accumulation_steps=3, gradient_clip=1.0,
        scheduler="cosine", scheduler_kwargs={"eta_min": 0.01},
        experiment_dir=tmp_path / "experiment")
    lr = result.history["learning_rate"]
    assert len(lr) == 4
    assert lr[0] == pytest.approx(0.12)
    assert all(a > b for a, b in zip(lr, lr[1:]))
    assert lr[-1] > 0.01
    path = tmp_path / "experiment"
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["config"]["accumulation_steps"] == 3
    assert manifest["config"]["gradient_clip"] == 1.0
    assert manifest["config"]["scheduler"] == "cosine"
    assert json.loads((path / "history.json").read_text())["learning_rate"] == lr
    assert (path / "best.pt").exists()
    assert len(load_checkpoint(result.model, path / "best.pt")["learning_rate"]) >= 1


def test_step_and_plateau_scheduler_work_and_record_lr():
    torch, data = _dataset()
    from lispyn.workflow import train

    step_run = train(
        torch.nn.Linear(2, 2), data, loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=3, batch_size=2, lr=0.2, device="cpu",
        scheduler="step", scheduler_kwargs={"step_size": 1, "gamma": 0.5})
    assert step_run.history["learning_rate"] == pytest.approx([0.2, 0.1, 0.05])

    plateau = train(
        torch.nn.Linear(2, 2), data,
        validation=data, loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=3, batch_size=2, lr=0.1, device="cpu",
        scheduler="plateau",
        scheduler_kwargs={"factor": 0.5, "patience": 0,
                          "threshold": 1000, "threshold_mode": "abs"})
    assert len(plateau.history["learning_rate"]) == 3
    assert plateau.history["learning_rate"][-1] < plateau.history["learning_rate"][0]


def test_scheduler_option_errors():
    torch, data = _dataset()
    from lispyn.workflow import train

    def run(**kwargs):
        return train(torch.nn.Linear(2, 2), data, device="cpu",
                     loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        run(scheduler="unknown")
    with pytest.raises(ValueError, match="requires a scheduler"):
        run(scheduler_kwargs={"gamma": 0.8})
    with pytest.raises(TypeError, match="scheduler"):
        run(scheduler=42)


def test_lr_figure_and_missing_lr_history(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from lispyn.plots import plot_learning_rate

    fig = plot_learning_rate(
        {"learning_rate": [0.2, 0.1, 0.05]},
        theme="docusaurus-dark", palette="turquoise-spectrum",
        save=tmp_path / "lr.svg", title="LR schedule",
        subtitle="Synthetic schedule")
    assert fig.axes[0].lines[0].get_ydata().tolist() == [0.2, 0.1, 0.05]
    assert to_rgba(fig.get_facecolor()) == to_rgba("#1B1B1D")
    assert to_rgba(fig.axes[0].lines[0].get_color()) == to_rgba("#40E0D0")
    assert (tmp_path / "lr.svg").stat().st_size > 500
    plt.close(fig)
    with pytest.raises(ValueError, match="No learning_rate"):
        plot_learning_rate({"train_loss": [1, 0.8]})


def test_real_hy_macros_and_generated_notebook(tmp_path):
    pytest.importorskip("torch")
    from lispyn.notebooks import build_notebook

    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "advanced_training.hy"
    book = build_notebook(source)
    code = "\n".join(cell.source for cell in book.cells
                     if cell.cell_type == "code")
    assert "accumulation_steps=3" in code
    assert "scheduler='cosine'" in code or 'scheduler="cosine"' in code
    assert "plot_learning_rate" in code
    assert sum(c.cell_type == "markdown" for c in book.cells) >= 3
    assert all(c.get("execution_count") is None
               for c in book.cells if c.cell_type == "code")

    hy = shutil.which("hy")
    assert hy
    tiny = tmp_path / "minimal.hy"
    tiny.write_text(
        '(require lispyn.macros [learning-rate-curve])\n'
        '(import matplotlib)\n'
        '(matplotlib.use "Agg")\n'
        '(setv fig (learning-rate-curve {"learning_rate" [0.1 0.05]} '
        ':save "curve.png"))\n'
        '(print "lr-macro-ok")\n',
        encoding="utf-8")
    result = subprocess.run([hy, str(tiny)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "lr-macro-ok" in result.stdout
    assert (tmp_path / "curve.png").stat().st_size > 1000
