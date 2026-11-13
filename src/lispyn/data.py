"""Opt-in dataset utilities for small, reproducible PyTorch experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import random


def read_dataset(path: str | Path, **kwargs):
    """Load CSV, JSON, JSON Lines or Parquet as a pandas DataFrame.

    pandas is optional: ``pip install -e '.[data]'``. Parquet also requires
    an installed backend (typically pyarrow). File data is not executed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas first: pip install -e '.[data]'") from exc

    path = Path(path).expanduser()
    readers = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".jsonl": pd.read_json,
        ".parquet": pd.read_parquet,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    if path.suffix.lower() == ".jsonl":
        kwargs.setdefault("lines", True)
    return reader(path, **kwargs)


def drop_missing(frame, *, columns: Sequence[str] | None = None):
    """Return a new DataFrame with incomplete rows removed."""
    return frame.dropna(subset=list(columns) if columns is not None else None).reset_index(drop=True)


def train_validation_split(frame, *, validation_fraction: float = 0.2,
                           seed: int = 42, shuffle: bool = True):
    """Reproducible row-level train/validation split; no sklearn needed.

    Splits at least one row into each partition. This is not stratified or
    group-aware: users must choose a different split for time-series or
    datasets that could leak information between groups.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two rows are needed for a train/validation split")
    shuffled = frame.sample(frac=1, random_state=seed) if shuffle else frame
    validation_rows = max(1, min(len(frame) - 1, int(round(len(frame) * validation_fraction))))
    return (
        shuffled.iloc[:-validation_rows].reset_index(drop=True),
        shuffled.iloc[-validation_rows:].reset_index(drop=True),
    )



def split_dataset(dataset, *, validation_fraction: float = 0.2,
                  seed: int = 42, stratify: bool = False, labels=None):
    """Split any indexable PyTorch dataset into two non-overlapping Subsets.

    Set stratify=True to read dataset.targets (ImageFolder, AudioFolder,
    labeled Lispyn text datasets). Or pass labels explicitly, which also
    enables stratification; e.g. labels=[0, 1, ...] for TensorDataset.

    A class must contain >=2 samples so both partitions contain that class.
    Splitting is sample-level, not group-, speaker- or subject-aware: choose
    a grouped split when images/audio/text from one source can leak across
    partitions. Per-class rounding can change the overall validation fraction.
    """
    from torch.utils.data import Subset

    if isinstance(validation_fraction, bool) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    count = len(dataset)
    if count < 2:
        raise ValueError("At least two samples are needed for a dataset split")
    rng = random.Random(seed)

    if stratify or labels is not None:
        if labels is None:
            labels = getattr(dataset, "targets", None)
        if labels is None:
            raise ValueError("Stratified split requires labels or dataset.targets")
        labels = list(labels)
        if len(labels) != count:
            raise ValueError("Number of labels must match dataset length")
        groups = {}
        try:
            for index, label in enumerate(labels):
                if hasattr(label, "item"):
                    label = label.item()
                groups.setdefault(label, []).append(index)
        except TypeError as exc:
            raise ValueError("Class labels must be hashable scalars") from exc

        train_indices, valid_indices = [], []
        for group in groups.values():
            if len(group) < 2:
                raise ValueError(
                    "Stratified split needs at least two samples per class"
                )
            rng.shuffle(group)
            n_valid = max(1, min(len(group) - 1,
                                 round(len(group) * validation_fraction)))
            valid_indices.extend(group[:n_valid])
            train_indices.extend(group[n_valid:])
        rng.shuffle(train_indices)
        rng.shuffle(valid_indices)
    else:
        indices = list(range(count))
        rng.shuffle(indices)
        n_valid = max(1, min(count - 1, round(count * validation_fraction)))
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


def to_tensor(values, *, dtype=None, device=None):
    """Convert values into a tensor using the real, unrestricted torch API."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch separately for your platform") from exc
    return torch.as_tensor(values, dtype=dtype, device=device)


def dataframe_to_dataset(frame, *, features: Sequence[str], target: str,
                         feature_dtype=None, target_dtype=None):
    """Turn numeric DataFrame columns into a PyTorch TensorDataset.

    Features default to torch.float32; target defaults to torch.float32 for
    regression. For classification, pass target_dtype=torch.long. Target is
    a 1-D tensor, and the original DataFrame is not modified.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch separately for your platform") from exc
    features = list(features)
    if not features:
        raise ValueError("features must include at least one column")
    if target in features:
        raise ValueError("target must not also be a feature (target leakage)")
    missing = [name for name in [*features, target] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    x = torch.as_tensor(
        frame[features].to_numpy(dtype="float32", copy=True),
        dtype=feature_dtype or torch.float32,
    )
    y = torch.as_tensor(
        frame[target].to_numpy(copy=True), dtype=target_dtype or torch.float32
    )
    return torch.utils.data.TensorDataset(x, y)


def make_dataloader(dataset, *, batch_size: int = 32, shuffle: bool = False,
                    seed: int | None = None, **kwargs):
    """Build a native PyTorch DataLoader, optionally with seeded shuffling.

    Arbitrary PyTorch DataLoader keyword options can pass through ``kwargs``.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch separately for your platform") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if seed is not None:
        if "generator" in kwargs:
            raise ValueError("Pass either seed or generator, not both")
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
