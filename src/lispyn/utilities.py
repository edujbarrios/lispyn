"""Small, dependency-free helpers for Hy experiment diagnostics.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
from typing import Iterator


def assert_close(actual: float, expected: float, *, rel_tol: float = 1e-9,
                 abs_tol: float = 0.0) -> bool:
    """Check two finite scalar results with explicit, nonnegative tolerances.

    Return True for convenient use in Hy expressions; raise AssertionError
    with the measured values if the check fails.
    """
    for name, value in (("actual", actual), ("expected", expected),
                        ("rel_tol", rel_tol), ("abs_tol", abs_tol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real numeric scalar (not bool)")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be nonnegative")
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AssertionError(
            f"values differ: actual={actual!r}, expected={expected!r}, "
            f"rel_tol={rel_tol!r}, abs_tol={abs_tol!r}"
        )
    return True


@contextmanager
def timing(label: str) -> Iterator[None]:
    """Report elapsed wall-clock time, including when the body raises."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("timing label must be a nonempty string")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.3f} s")
