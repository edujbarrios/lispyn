"""Dynamic experiment storytelling for Hy and standard Jupyter.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

Author narrative templates and aggregate scalar metrics in Hy/Lisp. Jupyter
displays the resulting Markdown; direct Hy execution prints the same text.
There is no expression evaluation or automatic sample/prediction capture.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
import re
from string import Formatter


_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER_FORMAT = re.compile(r"^(?:,)?(?:\.\d{1,2})?[eEfFgG%]?$")


def _scalar(value, label):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Real) and math.isfinite(value):
        return value
    raise ValueError(
        f"{label} must be a finite numeric, Boolean, string, or None scalar"
    )


def _markdown_cell(text):
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_narrative(template, **values):
    """Interpolate named scalar observations into authored Markdown.

    Only {name} and simple numeric specifications such as {accuracy:.1%}
    are supported. Dotted/indexed expressions, conversions, and nested
    formats are explicitly rejected. Neither Python nor Hy is evaluated
    from a template, and unused value arguments are rejected.
    """
    if not isinstance(template, str) or not template.strip():
        raise ValueError("template must be nonempty Markdown text")
    for key, value in values.items():
        if not _FIELD.fullmatch(key):
            raise ValueError(f"Invalid narrative field name: {key!r}")
        _scalar(value, key)

    seen = set()
    pieces = []
    try:
        segments = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"Invalid narrative template: {exc}") from exc
    for literal, field, spec, conversion in segments:
        pieces.append(literal)
        if field is None:
            continue
        if not _FIELD.fullmatch(field) or conversion is not None:
            raise ValueError(
                "Narrative fields must be simple names without attributes, "
                "indexes, conversions, or expressions"
            )
        if field not in values:
            raise ValueError(f"Missing narrative field: {field}")
        if not _NUMBER_FORMAT.fullmatch(spec):
            raise ValueError("Unsupported narrative numeric format")
        value = values[field]
        if spec and (isinstance(value, bool) or not isinstance(value, Real)):
            raise ValueError(
                f"Numeric formatting needs a finite real number: {field}"
            )
        try:
            pieces.append(format(value, spec))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Cannot format narrative field: {field}") from exc
        seen.add(field)
    unused = sorted(set(values) - seen)
    if unused:
        raise ValueError("Unused narrative values: " + ", ".join(unused))
    return "".join(pieces)


def render_metric_table(metrics, *, title="Experiment metrics", precision=5):
    """Return a Markdown summary table of explicit, finite scalar metrics."""
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a nonempty mapping")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be nonempty text")
    if isinstance(precision, bool) or not isinstance(precision, int) or not 1 <= precision <= 12:
        raise ValueError("precision must be an integer from 1 to 12")
    rows = [
        f"### {title.strip()}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric names must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Metric {name!r} must be a finite real number")
        rows.append(f"| {_markdown_cell(name)} | {value:.{precision}g} |")
    return "\n".join(rows) + "\n"


def _present(markdown, save):
    if save is not None:
        path = Path(save).expanduser()
        if path.suffix.lower() != ".md":
            raise ValueError("Narrative output must use a .md filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + ("" if markdown.endswith("\n") else "\n"),
                        encoding="utf-8")
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is not None and getattr(ipython, "kernel", None) is not None:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    else:
        print(markdown)
    return markdown


def show_narrative(template, *, save=None, **values):
    """Display runtime-interpolated Markdown in Jupyter or terminal Hy."""
    return _present(render_narrative(template, **values), save)


def show_metric_summary(metrics, *, title="Experiment metrics",
                        precision=5, save=None):
    """Display explicit aggregate metrics without accessing dataset contents."""
    return _present(render_metric_table(
        metrics, title=title, precision=precision,
    ), save)
