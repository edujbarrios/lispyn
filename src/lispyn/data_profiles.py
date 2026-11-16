"""Privacy-minimal NumPy and pandas data-quality summaries for Hy experiments.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
Returns and exports aggregate statistics only, not original data rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name must be a nonempty string")
    return value.strip()


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _publish(report, markdown, save):
    if save is not None:
        path = Path(save).expanduser()
        if path.suffix.lower() not in (".md", ".json"):
            raise ValueError("profile output must be .md or .json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n", encoding="utf-8")
        else:
            path.write_text(markdown + "\n", encoding="utf-8")
    from .narrative import _present
    _present(markdown, None)
    return report


def profile_array(values, *, name="Array", save=None):
    """Count finite values, NaN and inf, and summarize finite scalar values."""
    import numpy as np

    label = _name(name)
    arr = np.asarray(values)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise ValueError("array must be a nonempty 1-D or 2-D array")
    if arr.dtype.kind not in "biuf":
        raise TypeError("array must have real numeric or Boolean dtype")
    finite = arr[np.isfinite(arr)].astype(np.float64)
    size, n_finite = int(arr.size), int(finite.size)
    n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
    minimum = maximum = average = None
    if n_finite:
        minimum, maximum, average = map(float, (finite.min(), finite.max(), finite.mean()))
        if not all(map(math.isfinite, (minimum, maximum, average))):
            raise ValueError("finite array statistics overflowed float64")
    report = {
        "kind": "numpy", "name": label, "shape": list(map(int, arr.shape)),
        "dtype": str(arr.dtype), "total": size, "finite": n_finite,
        "nan": n_nan, "infinite": n_inf,
        "minimum": minimum, "maximum": maximum, "mean": average,
    }
    markdown = "\n".join([
        f"### {_cell(label)} — NumPy profile", "",
        "| Property | Value |", "| --- | ---: |",
        f"| Shape | {_cell(report['shape'])} |",
        f"| Data type | {_cell(report['dtype'])} |",
        f"| Elements | {size} |", f"| Finite | {n_finite} |",
        f"| NaN | {n_nan} |", f"| Infinity | {n_inf} |",
        f"| Min (finite only) | {minimum if minimum is not None else 'N/A'} |",
        f"| Max (finite only) | {maximum if maximum is not None else 'N/A'} |",
        f"| Mean (finite only) | {average if average is not None else 'N/A'} |",
    ])
    return _publish(report, markdown, save)


def profile_table(frame, *, name="Table", save=None, max_columns=100):
    """Count missingness, numeric infinities, duplicates and column dtypes.

    No sample rows are returned or printed. Tables are column-limited.
    """
    import numpy as np
    import pandas as pd

    label = _name(name)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if (isinstance(max_columns, bool) or not isinstance(max_columns, int)
            or not 1 <= max_columns <= 500):
        raise ValueError("max_columns must be an integer between 1 and 500")
    if frame.shape[1] == 0 or frame.shape[1] > max_columns:
        raise ValueError(f"table must have 1–{max_columns} columns")
    names = [str(column) for column in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique when converted to text")
    try:
        duplicates = int(frame.duplicated().sum())
    except TypeError as exc:
        raise ValueError("table columns must be hashable to count duplicates") from exc
    fields, missing_total, inf_total = [], 0, 0
    for key, text_name in zip(frame.columns, names):
        series = frame[key]
        missing = int(series.isna().sum())
        infinite = 0
        if (pd.api.types.is_numeric_dtype(series.dtype)
                and not pd.api.types.is_complex_dtype(series.dtype)):
            numbers = series.to_numpy(dtype="float64", na_value=np.nan)
            infinite = int(np.isinf(numbers).sum())
        missing_total += missing
        inf_total += infinite
        fields.append({"column": text_name, "dtype": str(series.dtype),
                       "missing": missing, "infinite": infinite})
    rows, cols = map(int, frame.shape)
    report = {"kind": "pandas", "name": label, "rows": rows, "columns": cols,
              "missing_cells": missing_total, "infinite_cells": inf_total,
              "duplicate_rows": duplicates, "fields": fields}
    markdown = "\n".join([
        f"### {_cell(label)} — pandas profile", "",
        f"Rows: **{rows}** · Columns: **{cols}** · Missing cells: **{missing_total}**"
        f" · Infinite numeric cells: **{inf_total}** · Duplicate rows: **{duplicates}**",
        "", "| Column | dtype | Missing | Infinite |",
        "| --- | --- | ---: | ---: |",
        *(f"| {_cell(field['column'])} | {_cell(field['dtype'])} | "
          f"{field['missing']} | {field['infinite']} |" for field in fields),
    ])
    return _publish(report, markdown, save)
