"""Small, opt-in dataset helpers for Lispyn experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path


def read_dataset(path: str | Path, **kwargs):
    """Read a CSV, JSON, or Parquet dataset into a pandas DataFrame.

    Pandas is an optional dependency: pip install -e '.[data]'.
    For other sources, call pandas, datasets, or your Python reader directly.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas first: pip install -e '.[data]'") from exc
    path = Path(path)
    methods = {".csv": pd.read_csv, ".json": pd.read_json, ".parquet": pd.read_parquet}
    reader = methods.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    return reader(path, **kwargs)


def to_tensor(values, *, dtype=None, device=None):
    """Convert numeric data to a tensor without restricting PyTorch's API."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch separately for your platform") from exc
    return torch.as_tensor(values, dtype=dtype, device=device)
