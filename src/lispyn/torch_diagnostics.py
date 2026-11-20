"""Opt-in, aggregate-only PyTorch diagnostics and inference helpers.

No model weights, tensor contents, or individual predictions are logged or
exported. PyTorch stays an optional, platform-selected dependency.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import itertools
import math

from .torch_utils import _require_module, move_to_device


def gradient_report(model):
    """Summarize dense gradients without modifying or retaining them.

    Norms are None when no gradients exist or any gradient is nonfinite.
    """
    import torch

    _require_module(model)
    parameters = list(model.parameters())
    present = 0
    elements = 0
    nonfinite = 0
    squares = 0.0
    maximum = 0.0
    with torch.no_grad():
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.layout != torch.strided:
                raise TypeError("gradient-report supports dense gradients only")
            present += 1
            elements += gradient.numel()
            finite = torch.isfinite(gradient)
            nonfinite += gradient.numel() - int(finite.sum().item())
            if bool(finite.all().item()) and gradient.numel():
                magnitude = gradient.detach().abs().to(torch.float64)
                squares += float(magnitude.square().sum().item())
                maximum = max(maximum, float(magnitude.max().item()))
    valid = nonfinite == 0 and math.isfinite(squares) and math.isfinite(maximum)
    return {
        "parameters": len(parameters),
        "parameters_with_grad": present,
        "gradient_elements": elements,
        "nonfinite_elements": nonfinite,
        "all_finite": bool(valid),
        "global_l2_norm": math.sqrt(squares) if valid and present else None,
        "max_absolute": maximum if valid and present else None,
    }


def assert_gradients(model, *, require_any=True):
    """Reject nonfinite dense gradients and, by default, missing gradients."""
    if not isinstance(require_any, bool):
        raise TypeError("require_any must be a bool")
    report = gradient_report(model)
    if require_any and not report["parameters_with_grad"]:
        raise ValueError("No gradients found; call backward() first")
    if not report["all_finite"]:
        raise ValueError(
            f"Nonfinite gradients detected: {report['nonfinite_elements']} elements"
        )
    return report


def model_footprint(model):
    """Nominal bytes of parameters/buffers, not peak allocation or activations."""
    _require_module(model)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
    return {
        "parameter_elements": sum(p.numel() for p in parameters),
        "buffer_elements": sum(b.numel() for b in buffers),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": parameter_bytes + buffer_bytes,
        "total_mib": round((parameter_bytes + buffer_bytes) / (1024 ** 2), 6),
    }


@contextmanager
def temporarily_freeze(module):
    """Restore every parameter's original requires_grad, even on error."""
    _require_module(module)
    parameters = [(p, p.requires_grad) for p in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, was_trainable in parameters:
            parameter.requires_grad_(was_trainable)


@contextmanager
def deterministic_algorithms(*, warn_only=False):
    """Temporarily enforce PyTorch's deterministic-algorithm policy.

    Does not set RNG seeds, CuDNN settings, or CUDA workspace environment vars.
    """
    import torch

    if not isinstance(warn_only, bool):
        raise TypeError("warn_only must be a bool")
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        yield
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)


def predict_batches(model, batches, *, device="cpu", max_batches=None,
                    input_key=None):
    """Return concatenated CPU logits from an iterable of inference batches.

    Batches may be tensors, (inputs, labels) tuples/lists, or dicts with an
    explicitly named input_key. Model may return a Tensor, object.logits, or
    a mapping with a 'logits' tensor. No labels are forwarded by default.
    Previous training/eval state is restored, even when inference fails.
    """
    import torch

    _require_module(model)
    if (max_batches is not None and
            (isinstance(max_batches, bool) or not isinstance(max_batches, int)
             or max_batches < 1)):
        raise ValueError("max_batches must be a positive integer or None")
    if input_key is not None and (not isinstance(input_key, str) or not input_key):
        raise ValueError("input_key must be a nonempty string or None")
    target = torch.device(device)
    # Preserve mixed train/eval flags (e.g., frozen BatchNorm layers).
    previous_modes = [(module, module.training) for module in model.modules()]
    results = []
    try:
        model.to(target)
        model.eval()
        with torch.inference_mode():
            source = itertools.islice(batches, max_batches) if max_batches else iter(batches)
            for batch in source:
                if isinstance(batch, torch.Tensor):
                    inputs = batch
                elif isinstance(batch, (tuple, list)) and len(batch) in (1, 2):
                    inputs = batch[0]
                elif isinstance(batch, Mapping):
                    if input_key is None:
                        raise ValueError("Mapping batches require an explicit input_key")
                    if input_key not in batch:
                        raise KeyError(input_key)
                    inputs = batch[input_key]
                else:
                    raise TypeError("Expected Tensor, (inputs, labels), or mapping batch")
                moved = move_to_device(inputs, target)
                if isinstance(moved, Mapping):
                    output = model(**moved)
                elif isinstance(moved, tuple):
                    output = model(*moved)
                else:
                    output = model(moved)
                if isinstance(output, Mapping):
                    output = output.get("logits")
                elif not isinstance(output, torch.Tensor):
                    output = getattr(output, "logits", None)
                if not isinstance(output, torch.Tensor) or output.ndim < 1:
                    raise TypeError("Model must return a batched Tensor or logits Tensor")
                if output.shape[0] == 0:
                    raise ValueError("Empty inference batch")
                results.append(output.detach().cpu())
        if not results:
            raise ValueError("Inference loader is empty")
        return torch.cat(results, dim=0)
    finally:
        for module, was_training in previous_modes:
            module.training = was_training
