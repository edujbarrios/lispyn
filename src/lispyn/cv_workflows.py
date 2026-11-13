"""Repeated cross-validation and metric reports for Hy-first experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Scoring is delegated to a callback that creates a fresh model per fold.
No fitted model, dataset row, prediction or label is saved in reports.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path

from .cross_validation import (
    _validate_seed,
    k_fold_indices,
    summarize_cv_scores,
)


def repeated_k_fold_indices(count, *, n_splits=5, n_repeats=2,
                            seed=42, shuffle=True, labels=None, groups=None):
    """Yield (repeat_index, fold_index, train_indices, validation_indices).

    Indices are zero-based. Within *each repeat* every record is in exactly
    one validation fold. Class stratification and whole-group boundaries
    follow k_fold_indices. Repeats have distinct deterministic RNG seeds.
    """
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    _validate_seed(seed)
    if n_repeats > 1 and not shuffle:
        raise ValueError("Repeated k-fold requires shuffle=True for distinct shuffles")
    # Materialize generators once so repeated rounds see the same labels.
    labels = None if labels is None else list(labels)
    groups = None if groups is None else list(groups)
    for repetition in range(n_repeats):
        for fold, (training, validation) in enumerate(
            k_fold_indices(
                count, n_splits=n_splits, seed=seed + repetition,
                shuffle=shuffle, labels=labels, groups=groups,
            )
        ):
            yield repetition, fold, training, validation


def repeated_k_fold_dataset(dataset, *, n_splits=5, n_repeats=2,
                            seed=42, shuffle=True, stratify=False,
                            labels=None, groups=None):
    """Yield (repeat, fold, train_subset, validation_subset) for PyTorch.

    Use `stratify=True` for `dataset.targets`, `labels=[...]` for custom
    classes or `groups=[...]` for group isolation. These options are
    mutually exclusive. All subsets refer to the original dataset.
    """
    from torch.utils.data import Subset

    if groups is not None and (stratify or labels is not None):
        raise ValueError("Choose grouped or stratified splitting, not both")
    if labels is None and stratify:
        labels = getattr(dataset, "targets", None)
        if labels is None:
            raise ValueError("Stratified split requires labels or dataset.targets")
    for repeat, fold, training, validation in repeated_k_fold_indices(
        len(dataset), n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, shuffle=shuffle, labels=labels, groups=groups,
    ):
        yield repeat, fold, Subset(dataset, training), Subset(dataset, validation)


def repeated_k_fold_split(frame, *, n_splits=5, n_repeats=2,
                          seed=42, shuffle=True, stratify=None, group=None):
    """Yield repeated (repeat, fold, train_frame, validation_frame) in Hy.

    Input DataFrame remains unchanged. The optional `stratify` and `group`
    arguments take column names (e.g. 'label' or 'subject_id').
    """
    from .data import _frame_partition_options

    labels, groups = _frame_partition_options(
        frame, stratify=stratify, group=group
    )
    for repeat, fold, training, validation in repeated_k_fold_indices(
        len(frame), n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, shuffle=shuffle, labels=labels, groups=groups,
    ):
        yield (repeat, fold,
               frame.iloc[training].reset_index(drop=True),
               frame.iloc[validation].reset_index(drop=True))


def _validate_scorer(scorer):
    if not callable(scorer):
        raise TypeError("scorer must be a callable accepting train, validation, fold, repeat")


def _metric_score(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Fold scorer must return a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Fold scorer must return a finite real number")
    return value


def _run_cv(folds, scorer, *, n_splits, n_repeats, metric, output):
    _validate_scorer(scorer)
    scores, rows, total_weight, weighted_sum = [], [], 0, 0.0
    for repetition, fold, training, validation in folds:
        # The caller owns model creation to avoid reusing fitted parameters.
        score = _metric_score(scorer(training, validation, fold, repetition))
        weight = len(validation)
        scores.append(score)
        weighted_sum += weight * score
        total_weight += weight
        rows.append({
            "repeat": repetition + 1,
            "fold": fold + 1,
            "train_size": len(training),
            "validation_size": weight,
            "score": score,
        })
    report = summarize_cv_scores(scores, metric=metric)
    report.update({
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "weighted_mean": weighted_sum / total_weight,
        "fold_results": rows,
    })
    if output is not None:
        save_cv_report(report, output)
    return report


def run_dataset_cross_validation(dataset, scorer, *, n_splits=5, n_repeats=1,
                                 seed=42, shuffle=True, stratify=False,
                                 labels=None, groups=None,
                                 metric="score", output=None):
    """Score fresh models on PyTorch folds with a Hy callback.

    `scorer(train_subset, validation_subset, fold, repeat)` must return a
    finite scalar; create a *new* model and optimizer inside that callback.
    `fold` and `repeat` are zero-based indices. No external held-out test
    data is ever passed to the scorer. `output` optionally writes JSON.
    """
    folds = repeated_k_fold_dataset(
        dataset, n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, shuffle=shuffle, stratify=stratify,
        labels=labels, groups=groups,
    )
    return _run_cv(folds, scorer, n_splits=n_splits,
                   n_repeats=n_repeats, metric=metric, output=output)


def run_frame_cross_validation(frame, scorer, *, n_splits=5, n_repeats=1,
                               seed=42, shuffle=True,
                               stratify=None, group=None,
                               metric="score", output=None):
    """Score fresh models on pandas folds through the same Hy callback API.

    `stratify` and `group` are DataFrame column names. Group boundaries
    remain isolated within every fold of each repeat.
    """
    folds = repeated_k_fold_split(
        frame, n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, shuffle=shuffle, stratify=stratify, group=group,
    )
    return _run_cv(folds, scorer, n_splits=n_splits,
                   n_repeats=n_repeats, metric=metric, output=output)


def save_cv_report(report, path):
    """Write a summary-only JSON report, never input samples/predictions."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".json":
        raise ValueError("Cross-validation reports must use a .json filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path
