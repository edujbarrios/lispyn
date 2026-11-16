"""Regression coverage for privacy-minimal NumPy and pandas Hy profiles."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "data_profiles.hy"


def test_numpy_profile_counts_missing_and_valid_values(tmp_path, capsys):
    np = pytest.importorskip("numpy")
    from lispyn.data_profiles import profile_array
    data = np.array([[1.0, np.nan], [3.0, np.inf]])
    saved = tmp_path / "array.json"
    result = profile_array(data, name="Readings", save=saved)
    assert result == json.loads(saved.read_text())
    assert result["shape"] == [2, 2] and result["dtype"] == "float64"
    assert result["nan"] == 1 and result["infinite"] == 1
    assert result["finite"] == 2 and result["mean"] == 2.0
    assert "Readings" in capsys.readouterr().out
    assert np.isnan(data[0, 1]), "input was modified"
    empty_values = profile_array(np.array([np.nan, -np.inf]))
    assert empty_values["finite"] == 0
    assert empty_values["mean"] is None
    for value, error in [
        (np.empty((0, 2)), ValueError),
        (np.ones((1, 1, 1)), ValueError),
        (np.array(["x"]), TypeError),
        (np.array([1 + 2j]), TypeError),
    ]:
        with pytest.raises(error):
            profile_array(value)


def test_pandas_profile_is_aggregated_and_does_not_mutate(tmp_path):
    pd = pytest.importorskip("pandas")
    from lispyn.data_profiles import profile_table
    frame = pd.DataFrame({
        "subject": ["PRIVATE1", "PRIVATE2", "PRIVATE2"],
        "measure|x": [1.0, float("inf"), float("inf")],
        "note": [None, None, None],
    })
    before = frame.copy(deep=True)
    dest = tmp_path / "profile.md"
    result = profile_table(frame, save=dest)
    pd.testing.assert_frame_equal(frame, before)
    assert result["rows"] == 3 and result["columns"] == 3
    assert result["missing_cells"] == 3
    assert result["infinite_cells"] == 2
    assert result["duplicate_rows"] == 1
    assert "PRIVATE1" not in dest.read_text()
    assert "PRIVATE2" not in json.dumps(result)
    assert "measure\|x" in dest.read_text()
    with pytest.raises(ValueError, match="columns"):
        profile_table(frame, max_columns=2)
    with pytest.raises(ValueError, match="column names"):
        profile_table(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(TypeError, match="DataFrame"):
        profile_table([1, 2])


def test_profile_paths_and_names(tmp_path):
    pd = pytest.importorskip("pandas")
    from lispyn.data_profiles import profile_array, profile_table
    for fn, data in ((profile_array, [1, 2]),
                     (profile_table, pd.DataFrame({"x": [1]}))):
        with pytest.raises(ValueError, match="name"):
            fn(data, name="")
        with pytest.raises(ValueError, match="output"):
            fn(data, save=tmp_path / "bad.txt")
        saved = tmp_path / (fn.__name__ + ".json")
        result = fn(data, save=saved)
        assert json.loads(saved.read_text()) == result


def test_hy_macros_work_in_direct_build_and_executed_notebook(tmp_path):
    pytest.importorskip("hy")
    pytest.importorskip("pandas")
    pytest.importorskip("nbclient")
    import nbformat
    from lispyn.notebooks import build_notebook

    built = build_notebook(EXAMPLE)
    nbformat.validate(built)
    assert any(c.cell_type == "markdown" for c in built.cells)
    code = "\n".join(c.source for c in built.cells if c.cell_type == "code")
    assert "lispyn.data_profiles.profile_array" in code
    assert "lispyn.data_profiles.profile_table" in code

    cli = [sys.executable, "-m", "lispyn.cli"]
    direct = subprocess.run(cli + ["run", str(EXAMPLE)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 0, direct.stderr
    assert "Data quality profiles complete." in direct.stdout
    unexecuted = tmp_path / "profile.ipynb"
    built_cli = subprocess.run(cli + ["build", str(EXAMPLE), "-o", str(unexecuted)],
                               cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert built_cli.returncode == 0, built_cli.stderr
    template = nbformat.read(unexecuted, as_version=4)
    assert all(c.execution_count is None and not c.outputs
               for c in template.cells if c.cell_type == "code")
    executed = tmp_path / "profile-executed.ipynb"
    execute = subprocess.run(cli + ["execute", str(EXAMPLE), "-o", str(executed),
                                    "--timeout", "90"],
                             cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert execute.returncode == 0, execute.stderr
    notebook = nbformat.read(executed, as_version=4)
    assert all(c.execution_count is not None for c in notebook.cells
               if c.cell_type == "code")
    outputs = [o for c in notebook.cells if c.cell_type == "code"
               for o in c.get("outputs", [])]
    stdout = "\n".join(str(o.get("text", "")) for o in outputs)
    markdown = "\n".join(str(o.get("data", {}).get("text/markdown", ""))
                         for o in outputs)
    assert "Data quality profiles complete." in stdout
    assert "NumPy profile" in markdown and "pandas profile" in markdown
