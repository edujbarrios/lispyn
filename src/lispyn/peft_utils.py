"""Concise PEFT/LoRA integration without restricting PyTorch or Transformers.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def lora(model, modules: Sequence[str], *, rank: int = 8,
         alpha: int | None = None, dropout: float = 0.05,
         task_type=None, **options):
    """Add Hugging Face PEFT LoRA adapters to an existing PyTorch model.

    Example: lora(model, ["q_lin", "v_lin"], rank=8, task_type="SEQ_CLS")
    The original model is modified in place by PEFT. Create the optimizer
    *after* this call, using only parameters where requires_grad is True.

    For plain torch.nn.Module models, omit task_type; PEFT compatibility
    depends on the base model's forward signature and target module types.
    Other LoraConfig options may be passed through **options.
    """
    if not modules or isinstance(modules, str):
        raise ValueError("modules must be a nonempty sequence of module names")
    if any(not isinstance(name, str) or not name.strip() for name in modules):
        raise ValueError("All module names must be nonempty strings")
    if rank < 1:
        raise ValueError("rank must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
    if alpha is not None and alpha < 1:
        raise ValueError("alpha must be positive")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install optional PEFT support: pip install -e '.[peft]'") from exc

    if isinstance(task_type, str):
        try:
            task_type = TaskType[task_type]
        except KeyError as exc:
            raise ValueError(f"Unknown PEFT task_type: {task_type}") from exc

    for reserved in ("r", "lora_alpha", "lora_dropout", "target_modules", "task_type"):
        if reserved in options:
            raise ValueError(f"Use Lispyn's named argument instead of {reserved!r}")
    config = LoraConfig(
        r=rank, lora_alpha=alpha if alpha is not None else 2 * rank,
        lora_dropout=dropout, target_modules=list(modules),
        task_type=task_type, **options,
    )
    return get_peft_model(model, config)


def parameter_counts(model) -> dict[str, int | float]:
    """Count trainable and total parameters in any torch.nn.Module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable": trainable,
        "total": total,
        "percent": (100 * trainable / total) if total else 0.0,
    }


def save_adapter(model, directory) -> Path:
    """Save PEFT adapter weights/config, not the full base model."""
    if not hasattr(model, "save_pretrained"):
        raise TypeError("Expected a PEFT model with save_pretrained()")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def load_adapter(base_model, directory, *, trainable=False):
    """Restore a locally saved PEFT adapter onto a compatible base model.

    Loading requires a trusted adapter directory. Pass trainable=True when
    resuming adapter training instead of using the model for inference.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Install optional PEFT support: pip install -e '.[peft]'") from exc
    return PeftModel.from_pretrained(
        base_model, str(directory), is_trainable=trainable,
    )
