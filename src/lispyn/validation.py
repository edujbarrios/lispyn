"""Lightweight validation guards for NumPy, pandas and PyTorch research inputs.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
These checks do not mutate or log input values.
"""
from __future__ import annotations

from numbers import Integral


def _label(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    return name.strip()


def assert_finite(values, *, name="values"):
    """Return values unchanged if all elements are finite.

    Supports real/complex NumPy arrays, numeric sequences, and optional
    PyTorch dense tensors. Errors report counts, not sensitive sample values.
    """
    label = _label(name)
    if type(values).__module__.split(".", 1)[0] == "torch":
        import torch
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{label} must be a numeric array or tensor")
        if values.numel() == 0:
            raise ValueError(f"{label} must not be empty")
        valid = torch.isfinite(values)
        invalid = int((~valid).count_nonzero().item())
        size = int(values.numel())
    else:
        import numpy as np
        arr = np.asarray(values)
        if arr.size == 0:
            raise ValueError(f"{label} must not be empty")
        if arr.dtype.kind not in "biufc":
            raise TypeError(f"{label} must contain numeric or Boolean values")
        valid = np.isfinite(arr)
        invalid = int(np.count_nonzero(~valid))
        size = int(arr.size)
    if invalid:
        raise AssertionError(
            f"{label} contains {invalid} non-finite value(s) out of {size}"
        )
    return values


def assert_shape(values, expected, *, name="values"):
    """Return a NumPy array, pandas DataFrame or torch Tensor unchanged.

    EXPECTED is 1–8 dimensions; None matches any dimension.
    """
    label = _label(name)
    actual = getattr(values, "shape", None)
    if actual is None:
        raise TypeError(f"{label} must expose a shape attribute")
    if not isinstance(expected, (tuple, list)) or not 1 <= len(expected) <= 8:
        raise ValueError("expected shape must be a list/tuple of 1 to 8 dimensions")
    for dimension in expected:
        if dimension is not None and (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or dimension < 0
        ):
            raise ValueError("expected shape dimensions must be nonnegative integers or None")
    try:
        shape = tuple(actual)
    except TypeError as exc:
        raise TypeError(f"{label} shape must be a sequence") from exc
    if len(shape) != len(expected) or any(
        want is not None and got != want for got, want in zip(shape, expected)
    ):
        raise AssertionError(
            f"{label} has shape {shape}, expected {tuple(expected)}"
        )
    return values
