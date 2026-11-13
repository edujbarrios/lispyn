"""Small, reusable PyTorch training, metrics and checkpoint helpers.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path


def trainable(model, *layers):
    """Freeze the model except named layer prefixes, or unfreeze all.

    trainable(model, 'fc') trains only ResNet's final classifier.
    trainable(model) enables full fine-tuning (all parameters).
    Create the optimizer *after* calling this function.
    """
    parameters = list(model.named_parameters())
    if layers and not any(
        name == layer or name.startswith(layer + ".")
        for name, _ in parameters for layer in layers
    ):
        raise ValueError(f"No parameters match layers: {layers!r}")
    for name, parameter in parameters:
        parameter.requires_grad_(
            not layers or any(name == layer or name.startswith(layer + ".") for layer in layers)
        )
    return model


def _move(value, device):
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_move(item, device) for item in value)
    return value.to(device) if hasattr(value, "to") else value


def accuracy(logits, labels):
    """Top-1 classification accuracy for logits [batch, classes]."""
    if logits.ndim != 2 or logits.shape[1] < 2 or labels.ndim != 1:
        raise ValueError("accuracy expects logits [batch, classes] and labels [batch]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels must have the same batch size")
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _metrics(metrics):
    if metrics is None:
        return {}
    if metrics == "accuracy":
        return {"accuracy": accuracy}
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be 'accuracy' or a mapping of metric names to callables")
    if any(not isinstance(k, str) or not k or k == "loss" or not callable(v)
           for k, v in metrics.items()):
        raise ValueError("Metric names must be nonempty strings (not 'loss') with callable values")
    return dict(metrics)


def _batch_result(model, batch, loss_fn, step, device):
    """One forward pass produces both the loss and optional prediction metrics."""
    batch = _move(batch, device)
    logits = labels = None
    if step is not None:
        loss = step(model, batch)
    elif isinstance(batch, Mapping):
        labels = batch.get("labels")
        output = model(**batch)
        logits = (output.logits if hasattr(output, "logits")
                  else output.get("logits") if isinstance(output, Mapping)
                  else None)
        if loss_fn is None:
            loss = (output.loss if hasattr(output, "loss")
                    else output["loss"])
        else:
            if labels is None:
                raise ValueError("Labeled mapping batch must contain 'labels'")
            if logits is None:
                raise ValueError("Model output must provide logits to use loss_fn")
            loss = loss_fn(logits, labels)
    else:
        if loss_fn is None:
            raise ValueError("Pass loss_fn for (inputs, labels) batches")
        inputs, labels = batch
        logits = model(inputs)
        loss = loss_fn(logits, labels)
    if getattr(loss, "ndim", None) != 0:
        raise ValueError("Training step must return a scalar loss tensor")
    return loss, logits, labels


def _batch_loss(model, batch, loss_fn, step, device):
    """Backward-compatible internal helper."""
    return _batch_result(model, batch, loss_fn, step, device)[0]


def _epoch(model, loader, *, loss_fn, step, device, metrics, optimizer=None):
    import torch

    if optimizer is None:
        model.eval()
    else:
        model.train()
    totals = {"loss": 0.0, **{name: 0.0 for name in metrics}}
    count = 0
    context = torch.no_grad() if optimizer is None else torch.enable_grad()
    with context:
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad()
            loss, logits, labels = _batch_result(model, batch, loss_fn, step, device)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach().item())
            if metrics:
                if logits is None or labels is None:
                    raise ValueError(
                        "Metrics need model logits and labels; "
                        "custom step(model, batch) only returns loss"
                    )
                with torch.no_grad():
                    for name, metric_fn in metrics.items():
                        value = metric_fn(logits.detach(), labels.detach())
                        totals[name] += float(value.item() if hasattr(value, "item") else value)
            count += 1
    if not count:
        raise ValueError("Training/validation loader is empty")
    return {name: total / count for name, total in totals.items()}


def evaluate_metrics(model, loader, *, loss_fn=None, metrics=None,
                     step=None, device="cpu"):
    """Evaluate and return {'loss': value, 'accuracy': ...} without gradients."""
    if step is not None and metrics is not None:
        raise ValueError("Custom step is loss-only; omit metrics or use a standard model output")
    model.to(device)
    return _epoch(model, loader, loss_fn=loss_fn, metrics=_metrics(metrics),
                  step=step, device=device)


def evaluate(model, loader, *, loss_fn=None, step=None, device="cpu"):
    """Mean validation loss per batch (backward-compatible float result)."""
    return evaluate_metrics(
        model, loader, loss_fn=loss_fn, step=step, device=device
    )["loss"]


def fit(model, loader, optimizer, *, loss_fn=None, epochs=1, device="cpu",
        validation=None, step=None, metrics=None, save_best=None,
        monitor=None, mode=None, log_path=None, patience=None,
        min_delta=0.0, save_last=None):
    """Train with optional metrics, JSON history and best checkpoint.

    - Standard batches: (inputs, labels) with loss_fn.
    - Transformers: mappings with 'labels', model output with 'loss' and 'logits'.
    - Custom step(model, batch): returns scalar loss, without extra metrics.
    - metrics='accuracy' or {'name': fn(logits, labels)}.
    - save_best='best.pt' uses minimum val_loss (or train_loss without validation).
      Set monitor='val_accuracy', mode='max' to select by accuracy.
    - Checkpoint saves the state at the *best epoch*; the in-memory model
      remains at the final epoch. Use load_checkpoint to restore the best.
    - patience=N stops after N consecutive epochs without an improvement
      exceeding min_delta on the selected monitor. No early stop by default.
    - save_last='last.pt' saves the latest model/optimizer after each epoch,
      allowing the last completed epoch to survive an interrupted run.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if patience is not None and (
        isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
    ):
        raise ValueError("patience must be a positive integer or None")
    if (isinstance(min_delta, bool) or not isinstance(min_delta, (int, float))
            or not math.isfinite(min_delta) or min_delta < 0):
        raise ValueError("min_delta must be a finite nonnegative number")
    if save_last is not None and save_best is not None and (
        Path(save_last).expanduser().resolve() == Path(save_best).expanduser().resolve()
    ):
        raise ValueError("save_last and save_best must use different paths")
    callbacks = _metrics(metrics)
    if step is not None and callbacks:
        raise ValueError("Custom step is loss-only; omit metrics or use standard model outputs")
    if mode not in (None, "min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    selection = monitor or ("val_loss" if validation is not None else "train_loss")
    if not (selection.startswith("train_") or selection.startswith("val_")):
        raise ValueError("monitor must start with train_ or val_")
    group, key = selection.split("_", 1)
    if key not in {"loss", *callbacks} or (group == "val" and validation is None):
        raise ValueError(f"Untracked monitor: {selection}")
    mode = mode or ("min" if key == "loss" else "max")

    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    for name in callbacks:
        history[f"train_{name}"] = []
        if validation is not None:
            history[f"val_{name}"] = []
    best = math.inf if mode == "min" else -math.inf
    stale_epochs = 0
    for _ in range(epochs):
        train = _epoch(model, loader, loss_fn=loss_fn, step=step, device=device,
                       metrics=callbacks, optimizer=optimizer)
        history["train_loss"].append(train["loss"])
        for name in callbacks:
            history[f"train_{name}"].append(train[name])
        if validation is not None:
            valid = _epoch(model, validation, loss_fn=loss_fn, step=step,
                           device=device, metrics=callbacks)
            history["val_loss"].append(valid["loss"])
            for name in callbacks:
                history[f"val_{name}"].append(valid[name])
        current = history[selection][-1]
        if not math.isfinite(current):
            raise ValueError(f"Non-finite {selection}: {current}")
        better = (
            current < best - min_delta if mode == "min"
            else current > best + min_delta
        )
        if better:
            best = current
            stale_epochs = 0
            if save_best is not None:
                save_checkpoint(model, save_best, optimizer=optimizer, history=history)
        else:
            stale_epochs += 1
        if save_last is not None:
            save_checkpoint(model, save_last, optimizer=optimizer, history=history)
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if patience is not None and stale_epochs >= patience:
            break
    return history


def _best_epoch(values, *, mode, min_delta=0.0):
    """Return (1-based epoch, value) matching early-stopping checkpoint selection."""
    best = math.inf if mode == "min" else -math.inf
    epoch = None
    for index, value in enumerate(values, 1):
        better = value < best - min_delta if mode == "min" else value > best + min_delta
        if better:
            epoch, best = index, value
    return epoch, best


def save_checkpoint(model, path, *, optimizer=None, history=None):
    """Save model (and optionally optimizer) using native torch state_dicts."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if history is not None:
        state["history"] = history
    torch.save(state, path)
    return path


def load_checkpoint(model, path, *, optimizer=None, map_location="cpu"):
    """Load a trusted state-dict checkpoint; weights_only is always enabled."""
    import torch

    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("history")
