"""Privacy-safe guards for partition leakage and malformed probabilities."""
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "evaluation_guards.hy"


def test_assert_disjoint_checks_real_identifiers_and_reports_counts_only():
    from lispyn.evaluation_guards import assert_disjoint

    assert assert_disjoint([1, 2, 2], [3, 4]) is True
    assert assert_disjoint(["1"], [1]) is True
    with pytest.raises(AssertionError, match=r"training and validation share 2 sample") as exc:
        assert_disjoint(["secret-a", "secret-b", "secret-b"],
                        ["secret-b", "secret-a", "private"])
    assert "secret" not in str(exc.value)
    with pytest.raises(AssertionError, match="train and test share 1"):
        assert_disjoint([1, 2], [2, 5], left_name="train", right_name="test")


@pytest.mark.parametrize("bad", [[], [True], [1.0], [None], [""], ["  "],
                                 [[1]], "abc", {"a": 1}])
def test_assert_disjoint_rejects_unstable_or_invalid_ids(bad):
    from lispyn.evaluation_guards import assert_disjoint

    with pytest.raises((TypeError, ValueError)):
        assert_disjoint(bad, [1])


def test_assert_disjoint_supports_numpy_and_pandas():
    np = pytest.importorskip("numpy")
    from lispyn.evaluation_guards import assert_disjoint

    assert assert_disjoint(np.array([1, 2]), np.array([3, 4]))
    with pytest.raises(ValueError, match="one-dimensional"):
        assert_disjoint(np.array([[1, 2]]), [3])
    pd = pytest.importorskip("pandas")
    assert assert_disjoint(pd.Series(["a", "b"]), pd.Series(["c"]))


def test_probability_guards_preserve_identity_and_require_normalized_rows():
    np = pytest.importorskip("numpy")
    from lispyn.evaluation_guards import assert_probabilities

    binary = np.array([0., 0.2, 1.])
    multiclass = np.array([[0.1, 0.9], [0.7, 0.3]])
    assert assert_probabilities(binary) is binary
    assert assert_probabilities(multiclass) is multiclass
    assert assert_probabilities([[0.4, 0.6], [0.2, 0.8]]) == [[0.4, 0.6], [0.2, 0.8]]
    with pytest.raises(AssertionError, match="2 invalid probability"):
        assert_probabilities(np.array([float("nan"), 1.2]))
    with pytest.raises(AssertionError, match="1 row"):
        assert_probabilities(np.array([[0.1, 0.2], [0.4, 0.6]]))
    assert assert_probabilities(np.array([[0.50001, 0.50001]]), atol=0.0001).shape == (1, 2)


@pytest.mark.parametrize("bad", [[], [0], [True], [1+0j], [[0.2]],
                                 [[[0.2, 0.8]]]])
def test_probability_guards_reject_wrong_type_or_shape(bad):
    from lispyn.evaluation_guards import assert_probabilities

    with pytest.raises((TypeError, ValueError)):
        assert_probabilities(bad)


@pytest.mark.parametrize("atol", [True, -1, float("nan"), float("inf"), 0.1, "0.1"])
def test_probability_guards_reject_bad_tolerance(atol):
    from lispyn.evaluation_guards import assert_probabilities

    with pytest.raises(ValueError, match="atol"):
        assert_probabilities([0.2], atol=atol)


def test_optional_torch_tensor_guards_preserve_gradients():
    torch = pytest.importorskip("torch")
    from lispyn.evaluation_guards import assert_disjoint, assert_probabilities

    values = torch.tensor([[0.25, 0.75]], requires_grad=True)
    assert assert_probabilities(values) is values
    assert values.requires_grad
    assert assert_disjoint(torch.tensor([1, 2]), torch.tensor([3, 4]))
    with pytest.raises(AssertionError, match="1 invalid probability"):
        assert_probabilities(torch.tensor([-0.1, 0.9]))
    with pytest.raises(AssertionError, match="1 row"):
        assert_probabilities(torch.tensor([[0.2, 0.2]]))


def test_hy_macros_compile_and_run_directly(tmp_path):
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook
    import nbformat

    notebook = build_notebook(EXAMPLE)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "lispyn.evaluation_guards.assert_disjoint" in code
    assert "lispyn.evaluation_guards.assert_probabilities" in code
    assert all(c.execution_count is None and not c.outputs
               for c in notebook.cells if c.cell_type == "code")

    cmd = [sys.executable, "-m", "lispyn.cli"]
    direct = subprocess.run(cmd + ["run", str(EXAMPLE)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert direct.returncode == 0, direct.stderr
    assert "Evaluation input guards passed." in direct.stdout
    output = tmp_path / "evaluation_guards.ipynb"
    executed = subprocess.run(cmd + ["execute", str(EXAMPLE), "-o", str(output),
                                "--timeout", "60"], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    assert executed.returncode == 0, executed.stderr
    rendered = nbformat.read(output, as_version=4)
    assert any("Evaluation input guards passed." in str(o.get("text", ""))
               for c in rendered.cells if c.cell_type == "code"
               for o in c.get("outputs", []))
