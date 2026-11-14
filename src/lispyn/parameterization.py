"""Reproducible, safe parameter overrides for notebooks authored in Hy.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.

The user writes (experiment-params (setv ...)) in Hy. Only the generated
Jupyter "parameters" code cells are changed; Lisp source is never rewritten.
Overrides are strict JSON literals, not Python/Hy code or expressions.
Compiling a *trusted* Hy source still invokes its compile-time macros.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import keyword
import math
from pathlib import Path
import re

from .compiler import LispynError
from .notebooks import write_notebook


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
MAX_VARIANTS = 64


def _identifier(name):
    """Normalize a common Hy name to its compiled Python identifier."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise LispynError(
            f"Invalid parameter name {name!r}; use letters, digits, _ or -"
        )
    compiled = name.replace("-", "_")
    if keyword.iskeyword(compiled) or not compiled.isidentifier():
        raise LispynError(f"Invalid parameter name {name!r}")
    return compiled


def _check_json(value):
    """Reject non-JSON types or non-finite numeric values."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LispynError("Parameter values must be finite JSON literals")
        return
    if isinstance(value, list):
        for element in value:
            _check_json(element)
        return
    if isinstance(value, dict):
        for key, element in value.items():
            if not isinstance(key, str):
                raise LispynError("Parameter object keys must be strings")
            _check_json(element)
        return
    raise LispynError("Parameter values must be JSON literals")


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(data):
    try:
        value = json.loads(
            data, object_pairs_hook=_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite number: {value}")
            ),
        )
        _check_json(value)
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise LispynError(f"Invalid JSON parameter value: {exc}") from exc


def load_json_object(path):
    """Read a strict, duplicate-free JSON object from disk."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".json":
        raise LispynError("Parameter configuration must be a .json file")
    value = _strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LispynError("Parameter configuration must be a JSON object")
    return value


def parse_set_args(assignments):
    """Parse --set NAME=JSON, rejecting duplicates and executable syntax."""
    result = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise LispynError("--set must be NAME=JSON (strings need quotes)")
        key, value = assignment.split("=", 1)
        name = _identifier(key)
        if name in result:
            raise LispynError(f"Duplicate --set parameter: {name}")
        result[name] = _strict_json(value)
    return result


def merge_overrides(config=None, assignments=()):
    """Merge JSON file values with CLI --set flags; --set wins."""
    result = {}
    for key, value in (config or {}).items():
        name = _identifier(key)
        if name in result:
            raise LispynError(f"Duplicate normalized parameter name: {name}")
        _check_json(value)
        result[name] = value
    result.update(parse_set_args(assignments))
    return result


def _parameters(notebook):
    """Collect all unambiguous top-level assignments in tagged code cells."""
    parameters = {}
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or "parameters" not in cell.metadata.get("tags", []):
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError as exc:
            raise LispynError(f"Invalid generated parameter code: {exc}") from exc
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            if name in parameters:
                raise LispynError(f"Duplicate notebook parameter: {name}")
            parameters[name] = (cell_index, node)
    return parameters


def list_parameters(notebook):
    """Describe parameter defaults without executing notebook code.

    For defaults that aren't a JSON-serializable literal, return the Python
    expression as text and mark literal=False. Never evaluate expressions.
    """
    details = []
    for name, (_, node) in _parameters(notebook).items():
        try:
            default = ast.literal_eval(node.value)
            _check_json(default)
        except (ValueError, TypeError, SyntaxError, MemoryError, LispynError):
            details.append({
                "name": name, "literal": False,
                "expression": ast.unparse(node.value),
            })
        else:
            details.append({
                "name": name, "literal": True, "default": default,
            })
    return details


def apply_overrides(notebook, overrides):
    """Change only declared parameters to JSON literal values in a notebook.

    Reject unknown names and ambiguous duplicate parameter declarations
    before making any edits. Notebook cell IDs are recalculated deterministically
    from the resulting source. Source-hash provenance remains unchanged.
    """
    parameters = _parameters(notebook)
    normalized = {}
    for key, value in overrides.items():
        name = _identifier(key)
        if name in normalized:
            raise LispynError(f"Duplicate normalized parameter name: {name}")
        _check_json(value)
        normalized[name] = value
    unknown = sorted(set(normalized).difference(parameters))
    if unknown:
        raise LispynError(
            "Unknown experiment parameter(s): " + ", ".join(unknown)
            + ". Run 'lispyn parameters SOURCE' to list supported names."
        )
    changed = {}
    for name, value in normalized.items():
        cell_index, _ = parameters[name]
        changed.setdefault(cell_index, {})[name] = value
    for cell_index, substitutions in changed.items():
        cell = notebook.cells[cell_index]
        tree = ast.parse(cell.source)
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in substitutions):
                # repr(JSON-literal) is valid Python; ast parsing ensures
                # that user-supplied strings are never executed as code.
                node.value = ast.parse(
                    repr(substitutions[node.targets[0].id]), mode="eval"
                ).body
        cell.source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        identity = (
            f"{notebook.metadata.lispyn.source}:{cell_index}:"
            f"{cell.cell_type}:{cell.source}"
        )
        cell.id = "lispyn-" + sha256(identity.encode()).hexdigest()[:16]
    if normalized:
        # Store names only: values already occur in code and may be sensitive.
        notebook.metadata.lispyn["overridden_parameters"] = sorted(normalized)
    return notebook


def build_sweep(notebook, grid, output_dir, *, max_variants=MAX_VARIANTS):
    """Compile a bounded Cartesian grid of parameterized, unexecuted notebooks.

    The source is built once. Variant notebooks only alter tagged parameter
    cells; no training is performed. The JSON index records the chosen values
    to make results reproducible, so review it before sharing.
    """
    if not isinstance(grid, dict) or not grid:
        raise LispynError("Sweep grid must be a nonempty JSON object")
    if (isinstance(max_variants, bool) or not isinstance(max_variants, int)
            or max_variants < 1):
        raise LispynError("max_variants must be a positive integer")
    available = set(_parameters(notebook))
    dimensions = []
    seen = set()
    total = 1
    for key, choices in grid.items():
        name = _identifier(key)
        if name in seen:
            raise LispynError(f"Duplicate normalized grid parameter: {name}")
        seen.add(name)
        if name not in available:
            raise LispynError(f"Unknown sweep parameter: {name}")
        if not isinstance(choices, list) or not choices:
            raise LispynError(f"Sweep values for {name} must be a nonempty array")
        for value in choices:
            _check_json(value)
        total *= len(choices)
        if total > max_variants:
            raise LispynError(
                f"Sweep exceeds {max_variants} variants; reduce the grid"
            )
        dimensions.append((name, choices))

    directory = Path(output_dir).expanduser().resolve()
    stem = Path(notebook.metadata.lispyn.source).stem
    outputs = [
        f"{stem}_variant_{index:03d}.ipynb"
        for index in range(1, total + 1)
    ]
    for filename in [*outputs, "sweep.json"]:
        if (directory / filename).exists():
            raise LispynError(
                f"Sweep would overwrite an existing file: {directory / filename}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": notebook.metadata.lispyn.source,
        "source_sha256": notebook.metadata.lispyn.get("source_sha256"),
        "variants": [],
    }
    for index, values in enumerate(product(*(choices for _, choices in dimensions))):
        overrides = dict(zip((name for name, _ in dimensions), values))
        variant = deepcopy(notebook)
        apply_overrides(variant, overrides)
        variant.metadata.lispyn["variant"] = index + 1
        filename = outputs[index]
        write_notebook(variant, directory / filename)
        manifest["variants"].append({
            "notebook": filename, "parameters": overrides,
        })
    (directory / "sweep.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return directory / "sweep.json"
