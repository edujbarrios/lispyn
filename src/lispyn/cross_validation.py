"""Leakage-safe three-way and k-fold data partitioning, without scikit-learn.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Only indices are partitioned: datasets and their samples are never copied.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, pstdev


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _validate_fraction(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < 1):
        raise ValueError(f"{name} must be strictly between 0 and 1")


def _three_sizes(count, validation_fraction, test_fraction):
    _validate_fraction(validation_fraction, "validation_fraction")
    _validate_fraction(test_fraction, "test_fraction")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    if count < 3:
        raise ValueError("At least three samples are needed for train/val/test")
    expected = [
        count * (1 - validation_fraction - test_fraction),
        count * validation_fraction,
        count * test_fraction,
    ]
    sizes = [max(1, math.floor(value)) for value in expected]
    # Largest-deficit allocation keeps the total exact and all partitions nonempty.
    while sum(sizes) < count:
        index = max(range(3), key=lambda i: (expected[i] - sizes[i], -i))
        sizes[index] += 1
    while sum(sizes) > count:
        eligible = [i for i in range(3) if sizes[i] > 1]
        index = min(eligible, key=lambda i: (expected[i] - sizes[i], i))
        sizes[index] -= 1
    return sizes


def _members(values, count, kind):
    if values is None:
        raise ValueError(f"{kind.capitalize()} values are required")
    members = {}
    values = list(values)
    if len(values) != count:
        raise ValueError(f"Number of {kind} values must match dataset length")
    for index, value in enumerate(values):
        if hasattr(value, "item"):
            value = value.item()
        try:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{kind.capitalize()} values cannot be non-finite")
            members.setdefault(value, []).append(index)
        except TypeError as exc:
            raise ValueError(f"{kind.capitalize()} values must be hashable scalars") from exc
    return list(members.values())


def _validate_modes(labels, groups):
    if labels is not None and groups is not None:
        raise ValueError("Choose either stratified or grouped splitting, not both")


def three_way_indices(count, *, validation_fraction=0.2, test_fraction=0.2,
                      seed=42, shuffle=True, labels=None, groups=None):
    """Return disjoint (train, validation, test) index lists.

    Stratification preserves every class across all three sets and requires
    at least three samples per class. Grouped mode keeps entire groups in
    one set and requires three distinct groups. The fractions are targets
    and may not be exact for tiny classes or indivisible groups.
    """
    _validate_seed(seed)
    _validate_modes(labels, groups)
    sizes = _three_sizes(count, validation_fraction, test_fraction)
    rng = random.Random(seed)
    partitions = [[], [], []]
    if groups is not None:
        units = _members(groups, count, "group")
        if len(units) < 3:
            raise ValueError("Three-way grouped split needs at least three distinct groups")
        if shuffle:
            rng.shuffle(units)
        units.sort(key=len, reverse=True)
        for number, unit in enumerate(units):
            empty = [i for i, bucket in enumerate(partitions) if not bucket]
            choices = empty if len(units) - number == len(empty) else range(3)
            choice = max(
                choices,
                key=lambda i: (sizes[i] - len(partitions[i]), -i),
            )
            partitions[choice].extend(unit)
    elif labels is not None:
        for unit in _members(labels, count, "class"):
            if len(unit) < 3:
                raise ValueError("Three-way stratification needs at least three samples per class")
            if shuffle:
                rng.shuffle(unit)
            size = _three_sizes(len(unit), validation_fraction, test_fraction)
            begin = 0
            for target, length in zip(partitions, size):
                target.extend(unit[begin:begin + length])
                begin += length
    else:
        indices = list(range(count))
        if shuffle:
            rng.shuffle(indices)
        start = 0
        for target, length in zip(partitions, sizes):
            target.extend(indices[start:start + length])
            start += length
    for bucket in partitions:
        if shuffle:
            rng.shuffle(bucket)
        else:
            bucket.sort()
    assert all(partitions), "internal error: empty data split"
    return tuple(partitions)


def k_fold_indices(count, *, n_splits=5, seed=42, shuffle=True,
                   labels=None, groups=None):
    """Yield (train_indices, validation_indices) with each sample held out once.

    With labels, each class must have >= n_splits examples. With groups,
    each group appears in exactly one validation fold and the number of
    distinct groups must be >= n_splits. Use a new model per fold; never
    tune hyperparameters on an external held-out test set.
    """
    _validate_seed(seed)
    _validate_modes(labels, groups)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if count < n_splits:
        raise ValueError("n_splits cannot exceed the dataset length")
    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    if groups is not None:
        units = _members(groups, count, "group")
        if len(units) < n_splits:
            raise ValueError("Grouped k-fold requires at least n_splits distinct groups")
        if shuffle:
            rng.shuffle(units)
        units.sort(key=len, reverse=True)
        for unit in units:
            index = min(range(n_splits), key=lambda i: (len(folds[i]), i))
            folds[index].extend(unit)
    elif labels is not None:
        offset = 0
        for unit in _members(labels, count, "class"):
            if len(unit) < n_splits:
                raise ValueError("Stratified k-fold requires at least n_splits samples per class")
            if shuffle:
                rng.shuffle(unit)
            for index, sample in enumerate(unit):
                folds[(offset + index) % n_splits].append(sample)
            offset += len(unit)
    else:
        indices = list(range(count))
        if shuffle:
            rng.shuffle(indices)
        for index, sample in enumerate(indices):
            folds[index % n_splits].append(sample)
    for bucket in folds:
        if shuffle:
            rng.shuffle(bucket)
        else:
            bucket.sort()
    for fold in range(n_splits):
        train = [sample for idx, bucket in enumerate(folds)
                 if idx != fold for sample in bucket]
        if not shuffle:
            train.sort()
        yield train, list(folds[fold])


def summarize_cv_scores(scores, *, metric="score"):
    """Summarize independent fold scores without storing predictions or samples.

    Returns count, mean, population standard deviation, minimum and maximum.
    Does not train a model or compute test scores on your behalf.
    """
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a nonempty string")
    values = list(scores)
    if not values:
        raise ValueError("At least one fold score is required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError("Fold scores must be finite real numbers")
    return {
        "metric": metric,
        "folds": len(values),
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }
