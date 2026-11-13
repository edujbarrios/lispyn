"""Chronological dataset splits and purged walk-forward cross-validation.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

No shuffling: training samples strictly precede each validation window.
A positional gap can prevent leakage from nearby observations. Input data
and pandas frames are never modified, and no scikit-learn is required.
"""

from __future__ import annotations

from .cross_validation import _three_sizes


def _int_arg(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def temporal_three_way_indices(count, *, validation_fraction=0.2,
                               test_fraction=0.2, gap=0):
    """Return chronological (train, validation, test) indices with two gaps.

    Fractions apply to *usable* samples after removing 2 * gap records.
    Gap records are deliberately excluded from every partition.
    The test set is last in time and should remain untouched until final
    evaluation. A gap is a number of samples, not a duration.
    """
    _int_arg(count, "count", 0)
    _int_arg(gap, "gap", 0)
    usable = count - 2 * gap
    if usable < 3:
        raise ValueError("Not enough samples for three nonempty time splits and two gaps")
    train_size, validation_size, test_size = _three_sizes(
        usable, validation_fraction, test_fraction
    )
    train_end = train_size
    validation_start = train_end + gap
    validation_end = validation_start + validation_size
    test_start = validation_end + gap
    return (list(range(train_end)),
            list(range(validation_start, validation_end)),
            list(range(test_start, test_start + test_size)))


def temporal_fold_indices(count, *, n_splits=5, test_size=None,
                          gap=0, min_train_size=1, max_train_size=None):
    """Yield (train_indices, validation_indices) in strictly ascending order.

    Walk forward across n_splits consecutive, non-overlapping validation
    windows ending at the last sample. Training data precede validation
    by at least gap positional samples. With max_train_size=None, the
    training window expands; otherwise it rolls over at that size.
    This is time-series CV: unlike k-fold, past samples are never tested
    again and an initial training prefix is never held out for validation.
    """
    _int_arg(count, "count", 1)
    _int_arg(n_splits, "n_splits", 2)
    _int_arg(gap, "gap", 0)
    _int_arg(min_train_size, "min_train_size", 1)
    if max_train_size is not None:
        _int_arg(max_train_size, "max_train_size", 1)
        if max_train_size < min_train_size:
            raise ValueError("max_train_size must be >= min_train_size")
    if test_size is None:
        test_size = count // (n_splits + 1)
    _int_arg(test_size, "test_size", 1)
    first_validation = count - n_splits * test_size
    if first_validation - gap < min_train_size:
        raise ValueError(
            "Not enough history: require n_splits * test_size + gap + "
            "min_train_size <= dataset length"
        )
    for fold in range(n_splits):
        start = first_validation + fold * test_size
        train_end = start - gap
        train_start = (0 if max_train_size is None
                       else max(0, train_end - max_train_size))
        yield (list(range(train_start, train_end)),
               list(range(start, start + test_size)))


def _ordered_frame(frame, time_column):
    """Create a chronologically ordered view-copy, validating unique times.

    Duplicate timestamps are rejected rather than splitting simultaneous
    observations across temporal boundaries. Without time_column, rows are
    assumed to already be in correct chronological order.
    """
    if time_column is None:
        return frame
    if not isinstance(time_column, str) or time_column not in frame.columns:
        raise ValueError(f"Missing time column: {time_column}")
    series = frame[time_column]
    if series.isna().any():
        raise ValueError(f"Time column {time_column!r} contains missing values")
    if not series.is_unique:
        raise ValueError("Time column must have unique timestamps to avoid leakage")
    try:
        return frame.sort_values(time_column, kind="stable")
    except (TypeError, ValueError) as exc:
        raise ValueError("Time column values must be mutually sortable") from exc


def temporal_three_way_frame(frame, *, validation_fraction=0.2,
                             test_fraction=0.2, gap=0, time_column=None):
    """Return chronologically disjoint DataFrames without modifying input.

    time_column sorts the frame first (stable ascending) and requires
    unique, nonmissing values. No random shuffling or stratification.
    """
    ordered = _ordered_frame(frame, time_column)
    partitions = temporal_three_way_indices(
        len(ordered), validation_fraction=validation_fraction,
        test_fraction=test_fraction, gap=gap,
    )
    return tuple(ordered.iloc[idx].reset_index(drop=True)
                 for idx in partitions)


def temporal_three_way_dataset(dataset, *, validation_fraction=0.2,
                               test_fraction=0.2, gap=0):
    """Return chronological PyTorch Subsets (train, validation, test)."""
    from torch.utils.data import Subset

    partitions = temporal_three_way_indices(
        len(dataset), validation_fraction=validation_fraction,
        test_fraction=test_fraction, gap=gap,
    )
    return tuple(Subset(dataset, idx) for idx in partitions)


def temporal_frame_folds(frame, *, n_splits=5, test_size=None, gap=0,
                         min_train_size=1, max_train_size=None,
                         time_column=None):
    """Yield chronological (train, validation) DataFrame windows."""
    ordered = _ordered_frame(frame, time_column)
    for training, validation in temporal_fold_indices(
        len(ordered), n_splits=n_splits, test_size=test_size,
        gap=gap, min_train_size=min_train_size,
        max_train_size=max_train_size,
    ):
        yield (ordered.iloc[training].reset_index(drop=True),
               ordered.iloc[validation].reset_index(drop=True))


def temporal_dataset_folds(dataset, *, n_splits=5, test_size=None, gap=0,
                           min_train_size=1, max_train_size=None):
    """Yield chronological (train, validation) native PyTorch Subsets."""
    from torch.utils.data import Subset

    for training, validation in temporal_fold_indices(
        len(dataset), n_splits=n_splits, test_size=test_size,
        gap=gap, min_train_size=min_train_size,
        max_train_size=max_train_size,
    ):
        yield Subset(dataset, training), Subset(dataset, validation)


def run_temporal_dataset_cv(dataset, scorer, *, n_splits=5, test_size=None,
                            gap=0, min_train_size=1, max_train_size=None,
                            metric="score", output=None):
    """Score fresh models per walk-forward fold using a Hy callback.

    The callback receives (train_subset, validation_subset, fold, repeat),
    with fold zero-based and repeat always zero. Keep any external test set
    separate. Reports contain scores/counts only, not raw observations.
    """
    from .cv_workflows import _run_cv

    folds = ((0, fold, train, valid) for fold, (train, valid) in enumerate(
        temporal_dataset_folds(
            dataset, n_splits=n_splits, test_size=test_size, gap=gap,
            min_train_size=min_train_size, max_train_size=max_train_size,
        )
    ))
    result = _run_cv(folds, scorer, n_splits=n_splits, n_repeats=1,
                     metric=metric, output=None)
    result.update({
        "strategy": "walk_forward",
        "gap": gap,
        "test_size": test_size if test_size is not None
                     else len(dataset) // (n_splits + 1),
        "min_train_size": min_train_size,
        "max_train_size": max_train_size,
    })
    if output is not None:
        from .cv_workflows import save_cv_report
        save_cv_report(result, output)
    return result


def run_temporal_frame_cv(frame, scorer, *, n_splits=5, test_size=None,
                          gap=0, min_train_size=1, max_train_size=None,
                          time_column=None, metric="score", output=None):
    """Score models on ordered DataFrame windows through a Hy callback."""
    from .cv_workflows import _run_cv

    folds = ((0, fold, train, valid) for fold, (train, valid) in enumerate(
        temporal_frame_folds(
            frame, n_splits=n_splits, test_size=test_size, gap=gap,
            min_train_size=min_train_size, max_train_size=max_train_size,
            time_column=time_column,
        )
    ))
    result = _run_cv(folds, scorer, n_splits=n_splits, n_repeats=1,
                     metric=metric, output=None)
    result.update({
        "strategy": "walk_forward",
        "gap": gap,
        "test_size": test_size if test_size is not None
                     else len(frame) // (n_splits + 1),
        "min_train_size": min_train_size,
        "max_train_size": max_train_size,
    })
    if output is not None:
        from .cv_workflows import save_cv_report
        save_cv_report(result, output)
    return result
