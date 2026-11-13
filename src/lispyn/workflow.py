"""Minimal orchestration for native PyTorch training.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

This module intentionally does not replace PyTorch: it selects a device,
optionally configures head-only/LoRA fine-tuning, creates native loaders and
an optimizer, and delegates the actual training to lispyn.training.fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from . import __version__
from .reproducibility import environment_info, seed_all, utc_now, write_json
from .training import fit, trainable


@dataclass
class Run:
    """Objects returned by train(); the model is the one actually trained."""

    model: Any
    history: dict[str, list[float]]
    optimizer: Any
    device: str
    manifest_path: Path | None = None


def device_auto() -> str:
    """Choose CUDA, MPS or CPU using PyTorch's backend availability checks."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _loader(data, *, batch_size, shuffle, seed):
    import torch

    if isinstance(data, torch.utils.data.DataLoader):
        return data
    if isinstance(data, torch.utils.data.Dataset):
        from .data import make_dataloader
        return make_dataloader(data, batch_size=batch_size, shuffle=shuffle, seed=seed)
    # An existing iterable of batches (e.g. a simple list of HF-style dicts)
    # should be passed through unchanged.
    if not hasattr(data, "__iter__"):
        raise TypeError("data must be a PyTorch Dataset, DataLoader or iterable of batches")
    return data


def train(model, data, *, loss_fn=None, validation=None, epochs=1, lr=1e-3,
          batch_size=32, optimizer="adamw", optimizer_kwargs=None,
          metrics=None, device="auto", seed=None, head=None,
          lora_modules=None, lora_rank=8, lora_task_type=None,
          step=None, save_best=None, log_path=None, monitor=None, mode=None,
          experiment_dir=None):
    """Train or fine-tune with one call, keeping native PyTorch escape hatches.

    - data and validation: torch Dataset, DataLoader or iterable of batches.
    - head='fc': freeze all but that named layer before making optimizer.
    - lora_modules=['query','value']: wrap model with official PEFT LoRA;
      do not combine head and lora_modules.
    - optimizer: 'adamw', 'adam', 'sgd' or any PyTorch optimizer class.
      optimizer_kwargs forwards extra constructor options.
    - loss_fn is required for tuple (inputs, labels) batches; Transformer
      mapping batches can provide their own scalar model loss.
    - experiment_dir='runs/demo' opt-in: best.pt, history.json and a
      privacy-conscious manifest.json with seed and dependency versions.
    - Returns a Run object with model, history, optimizer, device and manifest_path.
    """
    import torch

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not 0 < lr:
        raise ValueError("lr must be positive")
    if head is not None and lora_modules is not None:
        raise ValueError("Choose head-only or LoRA tuning, not both")
    if seed is not None:
        seed_all(seed)
    if device == "auto":
        device = device_auto()
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'auto', a device name or torch.device")

    if lora_modules is not None:
        from .peft_utils import lora
        model = lora(model, lora_modules, rank=lora_rank, task_type=lora_task_type)
    elif head is not None:
        trainable(model, head)

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters remain")
    choices = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if isinstance(optimizer, str):
        try:
            optimizer_class = choices[optimizer.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown optimizer: {optimizer}") from exc
    elif callable(optimizer):
        optimizer_class = optimizer
    else:
        raise TypeError("optimizer must be a name or optimizer class")
    options = dict(optimizer_kwargs or {})
    if "lr" in options:
        raise ValueError("Use lr= instead of optimizer_kwargs['lr']")
    opt = optimizer_class(parameters, lr=lr, **options)

    training = _loader(data, batch_size=batch_size, shuffle=True, seed=seed)
    valid = (None if validation is None else
             _loader(validation, batch_size=batch_size, shuffle=False, seed=None))
    manifest_path = None
    manifest = None
    if experiment_dir is not None:
        if not str(experiment_dir).strip():
            raise ValueError("experiment_dir cannot be blank")
        directory = Path(experiment_dir).expanduser()
        if save_best is None:
            save_best = directory / "best.pt"
        if log_path is None:
            log_path = directory / "history.json"
        manifest_path = directory / "manifest.json"
        metric_names = (
            [metrics] if isinstance(metrics, str) else
            list(metrics.keys()) if isinstance(metrics, Mapping) else []
        )
        tuning = "lora" if lora_modules is not None else (
            "head" if head is not None else "full"
        )
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_at": utc_now(),
            "lispyn_version": __version__,
            "model": type(model).__module__ + "." + type(model).__qualname__,
            "device": str(device),
            "environment": environment_info(),
            "config": {
                "seed": seed, "epochs": epochs, "lr": lr,
                "batch_size": batch_size,
                "optimizer": type(opt).__name__,
                "metrics": metric_names,
                "validation": valid is not None,
                "tuning": tuning,
                "head": head,
                "lora_modules": list(lora_modules) if lora_modules is not None else None,
                "lora_rank": lora_rank if lora_modules is not None else None,
            },
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters()
                                 if p.requires_grad),
            },
        }
        write_json(manifest_path, manifest)

    try:
        history = fit(
            model, training, opt, loss_fn=loss_fn, epochs=epochs, device=device,
            validation=valid, step=step, metrics=metrics, save_best=save_best,
            log_path=log_path, monitor=monitor, mode=mode,
        )
    except Exception as exc:
        if manifest is not None:
            manifest["status"] = "failed"
            manifest["failure_type"] = type(exc).__name__
            write_json(manifest_path, manifest)
        raise

    if manifest is not None:
        tracked = monitor or ("val_loss" if valid is not None else "train_loss")
        values = history[tracked]
        maximize = (mode == "max" if mode is not None
                    else not tracked.endswith("_loss"))
        best_value = max(values) if maximize else min(values)
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["summary"] = {
            "monitor": tracked,
            "mode": "max" if maximize else "min",
            "epoch": values.index(best_value) + 1,
            "value": best_value,
        }
        write_json(manifest_path, manifest)
    return Run(model=model, history=history, optimizer=opt,
               device=str(device), manifest_path=manifest_path)
