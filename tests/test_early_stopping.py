"""Early stopping, saved checkpoints, run manifest, and Hy macro integration."""

import json
import math
from pathlib import Path

import pytest


def test_early_stopping_respects_min_delta_and_saves_last_every_epoch(monkeypatch, tmp_path):
    import lispyn.training as training

    class DummyModel:
        def to(self, device):
            return self

    reports = iter([
        {"loss": 1.5}, {"loss": 1.0},
        {"loss": 1.4}, {"loss": 0.80},
        {"loss": 1.3}, {"loss": 0.78},
        {"loss": 1.2}, {"loss": 0.76},
    ])
    monkeypatch.setattr(training, "_epoch", lambda *a, **kw: next(reports))
    captured = []

    def record(model, path, *, optimizer=None, history=None):
        captured.append((Path(path).name, len(history["val_loss"])))
    monkeypatch.setattr(training, "save_checkpoint", record)

    log = tmp_path / "history.json"
    history = training.fit(
        DummyModel(), [], object(), validation=[], epochs=20,
        save_best=tmp_path / "best.pt", save_last=tmp_path / "last.pt",
        log_path=log, patience=2, min_delta=0.05,
    )
    assert history["val_loss"] == [1.0, 0.8, 0.78, 0.76]
    assert [ep for name, ep in captured if name == "best.pt"] == [1, 2]
    assert [ep for name, ep in captured if name == "last.pt"] == [1, 2, 3, 4]
    assert json.loads(log.read_text()) == history
    assert training._best_epoch(
        history["val_loss"], mode="min", min_delta=0.05
    ) == (2, 0.8)


def test_maximizing_accuracy_obeys_patience(monkeypatch):
    import lispyn.training as training

    class DummyModel:
        def to(self, device):
            return self

    reports = iter([
        {"loss": 2.0, "accuracy": 0.55},
        {"loss": 2.0, "accuracy": 0.70},
        {"loss": 2.0, "accuracy": 0.73},
        {"loss": 2.0, "accuracy": 0.74},
    ])
    monkeypatch.setattr(training, "_epoch", lambda *a, **kw: next(reports))
    history = training.fit(
        DummyModel(), [], object(), metrics={"accuracy": lambda x, y: 0.1},
        monitor="train_accuracy", mode="max", epochs=20,
        patience=2, min_delta=0.05,
    )
    assert history["train_accuracy"] == [0.55, 0.70, 0.73, 0.74]
    assert training._best_epoch(
        history["train_accuracy"], mode="max", min_delta=0.05
    ) == (2, 0.70)


@pytest.mark.parametrize("options, pattern", [
    ({"epochs": True}, "epochs"),
    ({"patience": 0}, "patience"),
    ({"patience": -1}, "patience"),
    ({"patience": True}, "patience"),
    ({"patience": 1.5}, "patience"),
    ({"min_delta": -0.1}, "min_delta"),
    ({"min_delta": True}, "min_delta"),
    ({"min_delta": math.nan}, "min_delta"),
    ({"min_delta": math.inf}, "min_delta"),
])
def test_invalid_stopping_configuration(options, pattern):
    from lispyn.training import fit
    with pytest.raises(ValueError, match=pattern):
        fit(None, None, None, **options)


def test_best_and_last_checkpoint_paths_must_differ(tmp_path):
    from lispyn.training import fit
    path = tmp_path / "same.pt"
    with pytest.raises(ValueError, match="different paths"):
        fit(None, None, None, save_best=path, save_last=path)


def test_one_call_stops_and_reports_actual_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.workflow import train
    from lispyn.training import load_checkpoint
    from lispyn.reproducibility import summarize_run

    torch.manual_seed(42)
    data = torch.utils.data.TensorDataset(
        torch.tensor([[0., 0.], [1., 1.], [1., 0.], [0., 1.]]),
        torch.tensor([0, 1, 1, 0]),
    )
    directory = tmp_path / "experiment"
    run = train(
        torch.nn.Linear(2, 2), data, validation=data,
        loss_fn=torch.nn.CrossEntropyLoss(), epochs=20, patience=2,
        min_delta=100.0, metrics="accuracy", batch_size=2,
        device="cpu", seed=42, experiment_dir=directory,
    )
    assert run.stopped_early is True
    assert len(run.history["train_loss"]) == 3
    assert (directory / "best.pt").is_file()
    assert (directory / "last.pt").is_file()
    assert len(load_checkpoint(run.model, directory / "last.pt")["train_loss"]) == 3

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["epochs_completed"] == 3
    assert manifest["summary"]["stopped_early"] is True
    assert manifest["summary"]["epoch"] == 1
    assert manifest["summary"]["value"] == pytest.approx(run.history["val_loss"][0])
    assert manifest["config"]["patience"] == 2
    assert manifest["config"]["min_delta"] == 100.0
    assert manifest["config"]["save_last"] is True
    assert "Completed epochs: 3/20 (early stopping)" in summarize_run(directory)


def test_training_without_patience_still_runs_every_epoch(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.workflow import train
    data = torch.utils.data.TensorDataset(
        torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([0, 1])
    )
    path = tmp_path / "last.pt"
    run = train(
        torch.nn.Linear(2, 2), data, epochs=2, save_last=path,
        loss_fn=torch.nn.CrossEntropyLoss(), device="cpu"
    )
    assert len(run.history["train_loss"]) == 2
    assert run.stopped_early is False
    assert path.is_file()


def test_macro_expands_in_generated_notebook():
    from lispyn.notebooks import build_notebook
    root = Path(__file__).resolve().parents[1]
    nb = build_notebook(root / "examples" / "early_stopping.hy")
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "patience=4" in code
    assert "min_delta=0.01" in code
    assert "stopped_early" in code
    assert sum(c.cell_type == "markdown" for c in nb.cells) >= 3
