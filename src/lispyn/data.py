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


def _group_split_indices(values, *, validation_fraction, rng, shuffle=True):
    """Partition complete groups rather than individual samples.

    Greedily select groups closest to the requested validation size, with
    deterministic tie-breaking from the local RNG. Both sides stay nonempty.
    """
    members = {}
    for index, label in enumerate(values):
        if hasattr(label, "item"):
            label = label.item()
        try:
            members.setdefault(label, []).append(index)
        except TypeError as exc:
            raise ValueError("Group labels must be hashable scalars") from exc
    if len(members) < 2:
        raise ValueError("Group-aware split needs at least two distinct groups")
    candidates = list(members.values())
    if shuffle:
        rng.shuffle(candidates)
    target = len(values) * validation_fraction
    selected = set()
    for indices in candidates:
        if (len(selected) + len(indices) < len(values)
                and abs(len(selected) + len(indices) - target)
                < abs(len(selected) - target)):
            selected.update(indices)
    if not selected:
        selected.update(min(candidates, key=lambda indices: abs(len(indices) - target)))
    train_indices = [i for i in range(len(values)) if i not in selected]
    valid_indices = [i for i in range(len(values)) if i in selected]
    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(valid_indices)
    return train_indices, valid_indices


def train_validation_split(frame, *, validation_fraction: float = 0.2,
                           seed: int = 42, shuffle: bool = True,
                           stratify: str | None = None,
                           group: str | None = None):
    """Split a DataFrame without sklearn, optionally balancing or grouping rows.

    stratify='label' keeps every class in both partitions (at least two
    records per class). group='subject_id' keeps all rows from each subject
    in one partition to prevent cross-partition leakage. These modes cannot
    be combined. Exact validation fractions are not guaranteed for either.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if len(frame) < 2:
        raise ValueError("At least two rows are needed for a train/validation split")
    if stratify is not None and group is not None:
        raise ValueError("Choose either stratify or group, not both")
    if stratify is not None or group is not None:
        column = stratify if stratify is not None else group
        if not isinstance(column, str) or column not in frame.columns:
            raise ValueError(f"Missing split column: {column}")
        if frame[column].isna().any():
            raise ValueError(f"Split column {column!r} contains missing values")
        rng = random.Random(seed)
        labels = frame[column].tolist()
        if group is not None:
            train_indices, valid_indices = _group_split_indices(
                labels, validation_fraction=validation_fraction, rng=rng,
                shuffle=shuffle,
            )
        else:
            classes = {}
            try:
                for index, label in enumerate(labels):
                    classes.setdefault(label, []).append(index)
            except TypeError as exc:
                raise ValueError("Class labels must be hashable scalars") from exc
            train_indices, valid_indices = [], []
            for indices in classes.values():
                if len(indices) < 2:
                    raise ValueError("Stratified split needs at least two samples per class")
                if shuffle:
                    rng.shuffle(indices)
                n_valid = max(1, min(len(indices) - 1,
                                     round(len(indices) * validation_fraction)))
                valid_indices.extend(indices[:n_valid])
                train_indices.extend(indices[n_valid:])
            if shuffle:
                rng.shuffle(train_indices)
                rng.shuffle(valid_indices)
            else:
                train_indices.sort()
                valid_indices.sort()
        return (frame.iloc[train_indices].reset_index(drop=True),
                frame.iloc[valid_indices].reset_index(drop=True))
    shuffled = frame.sample(frac=1, random_state=seed) if shuffle else frame
    validation_rows = max(1, min(len(frame) - 1, int(round(len(frame) * validation_fraction))))
    return (
        shuffled.iloc[:-validation_rows].reset_index(drop=True),
        shuffled.iloc[-validation_rows:].reset_index(drop=True),
    )


def split_dataset(dataset, *, validation_fraction: float = 0.2,
                  seed: int = 42, stratify: bool = False, labels=None,
                  groups=None):
    """Split any indexable PyTorch dataset into two non-overlapping Subsets.

    Set stratify=True to read dataset.targets (ImageFolder, AudioFolder,
    labeled Lispyn text datasets). Or pass labels explicitly, which also
    enables stratification; e.g. labels=[0, 1, ...] for TensorDataset.

    A class must contain >=2 samples so both partitions contain that class.
    Pass groups=[...] to keep all samples from each subject/session together;
    this prevents leakage and requires at least two distinct groups.
    Grouped and stratified modes are mutually exclusive.
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

    if groups is not None:
        if stratify or labels is not None:
            raise ValueError("Choose grouped or stratified splitting, not both")
        groups = list(groups)
        if len(groups) != count:
            raise ValueError("Number of groups must match dataset length")
        train_indices, valid_indices = _group_split_indices(
            groups, validation_fraction=validation_fraction, rng=rng
        )
        return Subset(dataset, train_indices), Subset(dataset, valid_indices)

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
