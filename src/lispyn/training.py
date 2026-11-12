"""Small PyTorch training and transfer-learning helpers.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def trainable(model, *layers):
    """Freeze the model except named layer prefixes, or unfreeze all.

    trainable(model, 'fc') trains only ResNet's final classifier.
    trainable(model) enables full fine-tuning (all parameters).
    Returns model so calls can be chained. Create optimizer *after* calling.
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


def _batch_loss(model, batch, loss_fn, step, device):
    batch = _move(batch, device)
    if step is not None:
        loss = step(model, batch)
    elif isinstance(batch, Mapping):
        # Natural fit for Hugging Face models whose forward returns .loss.
        output = model(**batch)
        if loss_fn is None:
            loss = output.loss if hasattr(output, "loss") else output["loss"]
        else:
            label = batch.get("labels")
            if label is None:
                raise ValueError("Labeled mapping batch must contain 'labels'")
            logits = (output.logits if hasattr(output, "logits") else
                      output["logits"] if isinstance(output, Mapping) else output)
            loss = loss_fn(logits, label)
    else:
        if loss_fn is None:
            raise ValueError("Pass loss_fn for (inputs, labels) batches")
        inputs, labels = batch
        loss = loss_fn(model(inputs), labels)
    if getattr(loss, "ndim", None) != 0:
        raise ValueError("Training step must return a scalar loss tensor")
    return loss


def evaluate(model, loader, *, loss_fn=None, step=None, device="cpu"):
    """Mean validation loss over batches; custom step(model, batch) optional."""
    import torch

    model.to(device)
    model.eval()
    total = count = 0
    with torch.no_grad():
        for batch in loader:
            total += float(_batch_loss(model, batch, loss_fn, step, device).item())
            count += 1
    if not count:
        raise ValueError("Validation loader is empty")
    return total / count


def fit(model, loader, optimizer, *, loss_fn=None, epochs=1, device="cpu",
        validation=None, step=None):
    """Train using native torch, returning history of per-batch mean losses.

    Expected batches are (inputs, labels), Hugging Face-style mappings with
    'labels', or anything supported by a custom step(model, batch) callable.
    Use trainable(model, 'head') before constructing an optimizer for PEFT-like
    head-only training; call trainable(model) for full fine-tuning.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        total = count = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, loss_fn, step, device)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
            count += 1
        if not count:
            raise ValueError("Training loader is empty")
        history["train_loss"].append(total / count)
        if validation is not None:
            history["val_loss"].append(
                evaluate(model, validation, loss_fn=loss_fn, step=step, device=device)
            )
    return history


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
