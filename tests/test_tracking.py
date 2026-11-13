"""Fast CPU tests for v0.5 metrics and best-checkpoint behavior."""

import json
import math

import pytest


def _classifier():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = torch.tensor([0, 1, 1, 0])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=2, shuffle=False
    )
    return torch, model, loader


def test_accuracy_and_shapes():
    torch, _, _ = _classifier()
    from lispyn.training import accuracy

    logits = torch.tensor([[0., 2.], [3., 1.], [0., 1.]])
    labels = torch.tensor([1, 1, 1])
    assert accuracy(logits, labels) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="logits"):
        accuracy(torch.rand(3), labels)
    with pytest.raises(ValueError, match="same batch size"):
        accuracy(logits, labels[:1])


def test_fit_metrics_best_checkpoint_and_json(tmp_path):
    _, model, loader = _classifier()
    from lispyn.training import fit, evaluate_metrics, load_checkpoint

    loss_fn = __import__("torch").nn.CrossEntropyLoss()
    checkpoint = tmp_path / "runs" / "best.pt"
    history_path = tmp_path / "runs" / "history.json"
    optimizer = __import__("torch").optim.SGD(model.parameters(), lr=0.04)

    history = fit(model, loader, optimizer, loss_fn=loss_fn, epochs=3,
                  validation=loader, metrics="accuracy", save_best=checkpoint,
                  log_path=history_path)
    assert list(history) == ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    assert all(len(values) == 3 for values in history.values())
    assert checkpoint.exists()
    assert json.loads(history_path.read_text()) == history
    assert all(0.0 <= x <= 1.0 for x in history["val_accuracy"])

    saved_model = __import__("torch").nn.Linear(2, 2)
    saved_history = load_checkpoint(saved_model, checkpoint)
    score = evaluate_metrics(saved_model, loader, loss_fn=loss_fn, metrics="accuracy")
    assert score["loss"] == pytest.approx(min(history["val_loss"]), abs=1e-6)
    assert len(saved_history["train_loss"]) in (1, 2, 3)


def test_monitor_max_accuracy_and_custom_metrics(tmp_path):
    torch, model, loader = _classifier()
    from lispyn.training import accuracy, evaluate, fit
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    path = tmp_path / "acc.pt"
    history = fit(model, loader, opt, loss_fn=torch.nn.CrossEntropyLoss(),
                  metrics={"top1": accuracy}, validation=loader,
                  save_best=path, monitor="val_top1", mode="max", epochs=2)
    assert path.exists()
    assert len(history["val_top1"]) == 2
    assert math.isfinite(evaluate(model, loader, loss_fn=torch.nn.CrossEntropyLoss()))


def test_hf_style_mapping_outputs_and_single_forward():
    torch, _, _ = _classifier()
    from lispyn.training import fit

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.calls = 0

        def forward(self, input_ids, labels):
            self.calls += 1
            logits = self.layer(input_ids.float())
            return {"loss": torch.nn.functional.cross_entropy(logits, labels),
                    "logits": logits}

    model = Classifier()
    batch = {"input_ids": torch.eye(2), "labels": torch.tensor([0, 1])}
    history = fit(model, [batch], torch.optim.SGD(model.parameters(), lr=0.01),
                  metrics="accuracy", epochs=2)
    assert model.calls == 2
    assert len(history["train_accuracy"]) == 2


def test_validation_of_metric_selection(tmp_path):
    torch, model, loader = _classifier()
    from lispyn.training import evaluate_metrics, fit

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()
    with pytest.raises(ValueError, match="Untracked monitor"):
        fit(model, loader, optimizer, loss_fn=loss, monitor="val_loss")
    with pytest.raises(ValueError, match="mode"):
        fit(model, loader, optimizer, loss_fn=loss, mode="largest")
    with pytest.raises(ValueError, match="metrics"):
        fit(model, loader, optimizer, loss_fn=loss, metrics=["accuracy"])
    with pytest.raises(ValueError, match="Custom step"):
        fit(model, loader, optimizer, step=lambda m, b: torch.tensor(1.0),
            metrics="accuracy")
    with pytest.raises(ValueError, match="Custom step"):
        evaluate_metrics(model, loader, step=lambda m, b: torch.tensor(1.0),
                         metrics="accuracy")


def test_checkpoint_works_without_validation(tmp_path):
    torch, model, loader = _classifier()
    from lispyn.training import fit

    checkpoint = tmp_path / "loss.pt"
    hist = fit(model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
               loss_fn=torch.nn.CrossEntropyLoss(), epochs=2,
               save_best=checkpoint)
    assert checkpoint.exists()
    assert len(hist["train_loss"]) == 2
    assert hist["val_loss"] == []
