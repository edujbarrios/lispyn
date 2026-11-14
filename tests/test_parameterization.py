"""Hy-first parameter variants, safe CLI overrides and reproducible sweeps.

These tests build real Hy notebooks; execution is opt-in and tested on a tiny
standard-library-only program, so no PyTorch or external datasets are needed.
"""

import json
import ast
from hashlib import sha256
from pathlib import Path

import nbformat
import pytest

from lispyn.cli import main
from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook
from lispyn.parameterization import (
    apply_overrides,
    build_sweep,
    list_parameters,
    load_json_object,
    merge_overrides,
    parse_set_args,
)


SOURCE = """(require lispyn.macros [md])
(md "# Hy experiment parameterization")
(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 3)
  (setv learning-rate 0.01)
  (setv label "baseline"))
(print seed epochs learning-rate label)
"""


def make_source(tmp_path):
    file = tmp_path / "experiment.hy"
    file.write_text(SOURCE, encoding="utf-8")
    return file


def _parameter_values(nb):
    return {p["name"]: p["default"] for p in list_parameters(nb)}


def test_parameter_listing_and_content_provenance(tmp_path, capsys):
    source = make_source(tmp_path)
    nb = build_notebook(source)
    nbformat.validate(nb)
    assert nb.metadata.lispyn.source_sha256 == sha256(SOURCE.encode()).hexdigest()
    assert _parameter_values(nb) == {
        "seed": 42, "epochs": 3, "learning_rate": .01, "label": "baseline",
    }
    assert main(["parameters", str(source), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["source"] == source.name
    assert data["source_sha256"] == nb.metadata.lispyn.source_sha256
    assert len(data["parameters"]) == 4
    assert main(["parameters", str(source)]) == 0
    assert "learning_rate = 0.01" in capsys.readouterr().out
    # Source is never rewritten.
    assert source.read_text() == SOURCE


def test_build_overrides_support_hyphenated_keys_and_json_types(tmp_path):
    source = make_source(tmp_path)
    before = source.read_bytes()
    config = tmp_path / "options.json"
    config.write_text('{"seed": 18, "epochs": 12, "label": "from file"}')
    target = tmp_path / "variant.ipynb"
    assert main([
        "build", str(source), "--params", str(config),
        "--set", "epochs=2", "--set", "learning-rate=0.125",
        "--set", 'label="overridden"',
        "-o", str(target),
    ]) == 0
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    assert _parameter_values(notebook) == {
        "seed": 18, "epochs": 2, "learning_rate": .125, "label": "overridden",
    }
    assert notebook.metadata.lispyn.overridden_parameters == [
        "epochs", "label", "learning_rate", "seed"
    ]
    assert notebook.metadata.lispyn.source_sha256 == sha256(before).hexdigest()
    assert source.read_bytes() == before
    assert all(c.get("execution_count") is None for c in notebook.cells if c.cell_type=="code")
    # Deterministic cell IDs across identical builds, different for variants.
    second = tmp_path / "variant2.ipynb"
    assert main([
        "build", str(source), "--params", str(config),
        "--set", "epochs=2", "--set", "learning-rate=0.125",
        "--set", 'label="overridden"', "-o", str(second),
    ]) == 0
    rebuilt = nbformat.read(second, as_version=4)
    assert [c.id for c in notebook.cells] == [c.id for c in rebuilt.cells]
    plain = build_notebook(source)
    tagged = [i for i, c in enumerate(notebook.cells)
              if "parameters" in c.metadata.get("tags", [])][0]
    assert notebook.cells[tagged].id != plain.cells[tagged].id


def test_execute_parameterized_hy_notebook_end_to_end(tmp_path):
    pytest.importorskip("nbclient")
    source = make_source(tmp_path)
    target = tmp_path / "executed.ipynb"
    result = main([
        "execute", str(source), "--set", "epochs=9",
        "--set", "learning-rate=0.75",
        "--set", 'label="variant"', "-o", str(target), "--timeout", "60",
    ])
    assert result == 0
    nb = nbformat.read(target, as_version=4)
    texts = [
        str(output.get("text", ""))
        for cell in nb.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ]
    assert any("42 9 0.75 variant" in txt for txt in texts)
    assert source.read_text() == SOURCE


def test_sweep_creates_deterministic_unexecuted_variants(tmp_path, capsys):
    source = make_source(tmp_path)
    grid = tmp_path / "grid.json"
    grid.write_text('{"epochs": [2, 4], "learning-rate": [0.001, 0.01]}')
    output = tmp_path / "variants"
    assert main([
        "sweep", str(source), "--grid", str(grid), "-o", str(output)
    ]) == 0
    manifest = json.loads((output / "sweep.json").read_text())
    assert manifest["source"] == source.name
    assert manifest["source_sha256"] == sha256(SOURCE.encode()).hexdigest()
    assert [entry["parameters"] for entry in manifest["variants"]] == [
        {"epochs": 2, "learning_rate": .001},
        {"epochs": 2, "learning_rate": .01},
        {"epochs": 4, "learning_rate": .001},
        {"epochs": 4, "learning_rate": .01},
    ]
    assert len(list(output.glob("*.ipynb"))) == 4
    for i, entry in enumerate(manifest["variants"], start=1):
        variant = nbformat.read(output / entry["notebook"], as_version=4)
        nbformat.validate(variant)
        assert variant.metadata.lispyn.variant == i
        assert variant.metadata.lispyn.source_sha256 == manifest["source_sha256"]
        assert variant.metadata.lispyn.overridden_parameters == [
            "epochs", "learning_rate",
        ]
        assert _parameter_values(variant)["seed"] == 42
        for name, value in entry["parameters"].items():
            assert _parameter_values(variant)[name] == value
        assert all(c.get("execution_count") is None for c in variant.cells if c.cell_type == "code")
    # Existing sweep files are never overwritten by accident.
    assert main(["sweep", str(source), "--grid", str(grid), "-o", str(output)]) == 1
    assert "overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("value, error", [
    ("seed=__import__('os').system('echo bad')", "Invalid JSON"),
    ("seed=NaN", "Invalid JSON"),
    ("seed=Infinity", "Invalid JSON"),
    ("epochs=nope", "Invalid JSON"),
    ("invalid?", "NAME=JSON"),
    ("9bad=1", "Invalid parameter name"),
    ("class=2", "Invalid parameter name"),
])
def test_set_rejects_expressions_and_invalid_names(value, error):
    with pytest.raises(LispynError, match=error):
        parse_set_args([value])


def test_duplicate_names_and_bad_json_files(tmp_path):
    with pytest.raises(LispynError, match="Duplicate --set"):
        parse_set_args(["seed=1", "seed=2"])
    with pytest.raises(LispynError, match="Duplicate normalized"):
        merge_overrides({"learning-rate": 1, "learning_rate": 2})
    with pytest.raises(LispynError, match="Duplicate"):
        load_json_object(_write(tmp_path / "dupe.json", '{"epochs":1,"epochs":2}'))
    with pytest.raises(LispynError, match="JSON object"):
        load_json_object(_write(tmp_path / "list.json", '[1,2]'))
    with pytest.raises(LispynError, match=".json"):
        load_json_object(_write(tmp_path / "bad.txt", '{"x":1}'))


def _write(path, content):
    path.write_text(content)
    return path


def test_unknown_parameters_and_override_validation(tmp_path, capsys):
    source = make_source(tmp_path)
    nb = build_notebook(source)
    original = [c.source for c in nb.cells]
    with pytest.raises(LispynError, match="Unknown experiment parameter"):
        apply_overrides(nb, {"not-declared": 12})
    assert [c.source for c in nb.cells] == original
    with pytest.raises(LispynError, match="finite JSON"):
        apply_overrides(nb, {"epochs": float("nan")})
    with pytest.raises(LispynError, match="JSON literals"):
        apply_overrides(nb, {"epochs": object()})
    assert main(["build", str(source), "--set", "not-declared=5"]) == 1
    assert "Unknown experiment parameter" in capsys.readouterr().err


def test_arbitrary_expressions_are_displayed_not_evaluated(tmp_path):
    file = tmp_path / "expressions.hy"
    file.write_text(
        '(require lispyn.macros [experiment-params])\n'
        '(experiment-params (setv epochs (+ 1 2)))\n'
    )
    params = list_parameters(build_notebook(file))
    assert params == [{"name": "epochs", "literal": False, "expression": "1 + 2"}]


def test_duplicates_across_parameter_cells_rejected(tmp_path):
    file = tmp_path / "double.hy"
    file.write_text(
        '(require lispyn.macros [experiment-params])\n'
        '(experiment-params (setv epochs 1))\n'
        '(require lispyn.macros [experiment-params])\n'
        '(experiment-params (setv epochs 2))\n'
    )
    with pytest.raises(LispynError, match="Duplicate notebook parameter"):
        list_parameters(build_notebook(file))


def test_empty_or_oversized_sweeps_rejected_before_writing(tmp_path, capsys):
    source = make_source(tmp_path)
    for i, grid in enumerate([
        {},
        {"epochs": []},
        {"unknown": [1]},
        {"epochs": list(range(65))},
    ]):
        path = tmp_path / f"grid{i}.json"
        path.write_text(json.dumps(grid))
        out = tmp_path / f"out{i}"
        assert main(["sweep", str(source), "--grid", str(path), "-o", str(out)]) == 1
        assert not out.exists()
    assert main(["parameters", str(_write(tmp_path / "plain.hy", "(print 2)\n"))]) == 0
    assert "No experiment parameters" in capsys.readouterr().out


def test_legacy_parameter_tag_supported(tmp_path):
    source = tmp_path / "legacy.hy"
    source.write_text(
        '(require lispyn.macros [cell-tags])\n'
        '(cell-tags "parameters")\n'
        '(setv seed 42)\n'
    )
    notebook = build_notebook(source)
    assert list_parameters(notebook)[0]["name"] == "seed"
    apply_overrides(notebook, {"seed": 10})
    assert _parameter_values(notebook)["seed"] == 10
