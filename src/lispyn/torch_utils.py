"""Opt-in PyTorch utilities used by Lispyn's Hy macros.

Keep PyTorch a platform-specific optional dependency. These functions operate
on ordinary torch.nn.Module objects and tensors; they do not own a training loop.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real


def _require_module(module):
    import torch

    if not isinstance(module, torch.nn.Module):
        raise TypeError("Expected a torch.nn.Module")
    return module


def set_module_trainable(module, enabled: bool):
    """Set all parameters' requires_grad flags; do not alter train/eval mode.

    Rebuild or configure the optimizer after changing trainable parameters.
    """
    _require_module(module)
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    module.requires_grad_(enabled)
    return module


def parameter_counts(module) -> dict[str, int]:
    """Count unique model parameters by element, including frozen parameters."""
    _require_module(module)
    parameters = list(module.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def clip_gradients(module, max_norm, *, norm_type=2.0) -> float:
    """Clip existing gradients in-place and return the pre-clipping norm.

    Call after backward() and before optimizer.step(). When using GradScaler,
    unscale the optimizer's gradients before calling this helper.
    """
    import torch

    _require_module(module)
    if (isinstance(max_norm, bool) or not isinstance(max_norm, Real)
            or not math.isfinite(float(max_norm)) or max_norm <= 0):
        raise ValueError("max_norm must be a positive finite real number")
    if (isinstance(norm_type, bool) or not isinstance(norm_type, Real)
            or math.isnan(float(norm_type)) or norm_type <= 0):
        raise ValueError("norm_type must be positive (infinity is allowed)")
    gradients = [parameter for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        raise ValueError("No gradients to clip; call backward() first")
    total_norm = torch.nn.utils.clip_grad_norm_(
        gradients, float(max_norm), norm_type=float(norm_type), error_if_nonfinite=True
    )
    return float(total_norm)


def move_to_device(value, device, *, non_blocking=False):
    """Move tensors inside nested dict/list/tuple batches without mutating them.

    Nontensor values are left untouched. Mappings become plain dictionaries,
    while ordinary tuples and namedtuples retain their types.
    """
    import torch

    if not isinstance(non_blocking, bool):
        raise TypeError("non_blocking must be a bool")
    target = torch.device(device)

    def move(item):
        if isinstance(item, torch.Tensor):
            return item.to(target, non_blocking=non_blocking)
        if isinstance(item, Mapping):
            return {key: move(child) for key, child in item.items()}
        if isinstance(item, list):
            return [move(child) for child in item]
        if isinstance(item, tuple) and hasattr(item, "_fields"):
            return type(item)(*(move(child) for child in item))
        if isinstance(item, tuple):
            return tuple(move(child) for child in item)
        return item

    return move(value)
