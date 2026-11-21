"""Privacy-preserving evaluation input guards for Hy research notebooks.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
These checks never modify, print or export sample identifiers or predictions.
"""
from __future__ import annotations

import math
from numbers import Real

from .validation import _label


def _identifier_set(values, name):
    if (isinstance(values, (str, bytes, dict))
            or not hasattr(values, "__len__")
            or not hasattr(values, "__iter__")):
        raise TypeError(f"{name} must be a one-dimensional sequence of sample IDs")
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 0 < len(values) <= 1_000_000:
        raise ValueError(f"{name} must contain between 1 and 1,000,000 sample IDs")
    identifiers = set()
    for item in values:
        if hasattr(item, "item"):
            try:
                item = item.item()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"{name} must contain scalar IDs") from exc
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise TypeError(f"{name} IDs must be strings or integers (not booleans)")
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"{name} IDs must be nonempty strings")
            identifiers.add(("str", item))
        else:
            identifiers.add(("int", int(item)))
    return identifiers


def assert_disjoint(left, right, *, left_name="training", right_name="validation"):
    """Reject overlapping sample IDs across two evaluation partitions.

    Accepts lists, tuples, NumPy arrays, pandas Series and PyTorch 1D tensors.
    IDs must be strings or integers. Does not infer IDs from row content.
    """
    left_name, right_name = _label(left_name), _label(right_name)
    left_ids = _identifier_set(left, left_name)
    right_ids = _identifier_set(right, right_name)
    overlap = len(left_ids.intersection(right_ids))
    if overlap:
        raise AssertionError(
            f"{left_name} and {right_name} share {overlap} sample identifier(s)"
        )
    return True


def assert_probabilities(values, *, name="probabilities", atol=1e-6):
    """Validate 1D binary scores or 2D normalized multiclass probabilities.

    Returns the original values unchanged; does not clamp or normalize.
    For multiclass arrays, every row must sum to one within atol.
    """
    label = _label(name)
    if (isinstance(atol, bool) or not isinstance(atol, Real)
            or not math.isfinite(atol) or not 0 <= atol <= 0.01):
        raise ValueError("atol must be a finite number between 0 and 0.01")

    if type(values).__module__.split(".", 1)[0] == "torch":
        import torch
        if not isinstance(values, torch.Tensor) or values.layout != torch.strided:
            raise TypeError(f"{label} must be a dense numeric array or tensor")
        if not torch.is_floating_point(values):
            raise TypeError(f"{label} must contain floating-point probabilities")
        shape, ndim = tuple(values.shape), values.ndim
        if ndim not in (1, 2) or not shape[0] or (ndim == 2 and shape[1] < 2):
            raise ValueError(f"{label} must have shape (N,) or (N, C) with N > 0 and C >= 2")
        invalid = int((~torch.isfinite(values) | (values < 0) | (values > 1)).sum().item())
        if invalid:
            raise AssertionError(f"{label} contains {invalid} invalid probability value(s)")
        if ndim == 2:
            sum_dtype = (torch.float32 if values.dtype in (torch.float16, torch.bfloat16)
                         else values.dtype)
            bad_rows = int(((values.sum(dim=1, dtype=sum_dtype) - 1).abs() > atol).sum().item())
            if bad_rows:
                raise AssertionError(f"{label} has {bad_rows} row(s) that do not sum to 1")
    else:
        import numpy as np
        array = np.asarray(values)
        if array.dtype.kind != "f":
            raise TypeError(f"{label} must contain floating-point probabilities")
        shape, ndim = array.shape, array.ndim
        if ndim not in (1, 2) or not shape[0] or (ndim == 2 and shape[1] < 2):
            raise ValueError(f"{label} must have shape (N,) or (N, C) with N > 0 and C >= 2")
        invalid = int(np.count_nonzero(~np.isfinite(array) | (array < 0) | (array > 1)))
        if invalid:
            raise AssertionError(f"{label} contains {invalid} invalid probability value(s)")
        if ndim == 2:
            bad_rows = int(np.count_nonzero(np.abs(array.astype("float64").sum(axis=1) - 1) > atol))
            if bad_rows:
                raise AssertionError(f"{label} has {bad_rows} row(s) that do not sum to 1")
    return values
