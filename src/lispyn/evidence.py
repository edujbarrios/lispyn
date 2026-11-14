"""Human-readable evidence checks for experiments authored in Hy.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

No global state, model interception or metric collection: a Hy author supplies
explicit Boolean conditions, so claims are inspectable and reproducible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from numbers import Real


def _field(value, name, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _scalar(value, name):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    raise ValueError(f"{name} must be a finite JSON-compatible scalar")


def _save(value, path):
    if path is None:
        return
    path = Path(path).expanduser()
    if path.suffix.lower() != ".json":
        raise ValueError("Evidence reports must have a .json extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _display(text):
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is not None and getattr(ipython, "kernel", None) is not None:
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def experiment_check(label, condition, *, observed=None, expected=None,
                     notes=None, strict=False, save=None):
    """Evaluate an explicitly supplied Boolean, and display the evidence.

    Return a small JSON-compatible record with check, passed, observed,
    expected and notes. Only scalar observations are supported: no data rows,
    tensors, model parameters or predictions are captured automatically.
    If strict=True, write/display failure first, then raise AssertionError.
    """
    label = _field(label, "label")
    if not isinstance(condition, bool):
        raise TypeError("condition must be a Boolean comparison (use .item() for scalar tensors)")
    if not isinstance(strict, bool):
        raise TypeError("strict must be Boolean")
    expected = _scalar(expected, "expected")
    observed = _scalar(observed, "observed")
    if notes is not None:
        notes = _field(notes, "notes")
    result = {"check": label, "passed": condition}
    if observed is not None:
        result["observed"] = observed
    if expected is not None:
        result["expected"] = expected
    if notes is not None:
        result["notes"] = notes
    # Validate path and write before emitting the human-readable result.
    _save(result, save)
    status = "PASS" if condition else "FAIL"
    lines = [f"**[{status}] {_cell(label)}**"]
    if observed is not None or expected is not None:
        lines.append(
            f"Observed: {_cell(observed) if observed is not None else '—'}; "
            f"expected: {_cell(expected) if expected is not None else '—'}"
        )
    if notes is not None:
        lines.append(f"Notes: {_cell(notes)}")
    _display("\n\n".join(lines))
    if strict and not condition:
        raise AssertionError(f"Experiment check failed: {label}")
    return result


def checklist_report(checks, *, title="Experiment evidence", strict=False, save=None):
    """Render a reproducible evidence checklist from Hy check results.

    The report never reads source data. Failing checks are visible even
    when strict=False; strict=True raises after displaying and saving.
    """
    title = _field(title, "title")
    if not isinstance(strict, bool):
        raise TypeError("strict must be Boolean")
    rows = list(checks)
    if not rows:
        raise ValueError("checks must contain at least one experiment check")
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("check"), str) or not isinstance(row.get("passed"), bool):
            raise ValueError("Each check must contain a text 'check' and Boolean 'passed'")
        normalized.append(dict(row))
    passed = sum(row["passed"] for row in normalized)
    report = {"title": title, "passed": passed, "failed": len(normalized) - passed,
              "checks": normalized}
    _save(report, save)
    lines = [f"## {title}", "", f"**Passed:** {passed}/{len(normalized)}", ""]
    for row in normalized:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- **{status}** — {_cell(row['check'])}")
    _display("\n".join(lines))
    if strict and passed != len(normalized):
        raise AssertionError(f"{len(normalized) - passed} experiment check(s) failed")
    return report


# Hy macro-expansion targets intentionally differ from the public macro
# names. Otherwise importing a same-named function while expanding a Hy
# macro can recursively re-expand that very macro.
evaluate_check = experiment_check
summarize_evidence = checklist_report
