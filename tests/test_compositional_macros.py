"""Native Hy evaluation/prediction/iteration/RNG/gradient regression tests."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from lispyn.notebooks import build_notebook

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "macro_composition.hy"


def test_composition_expands_to_native_python():
    notebook = build_notebook(EXAMPLE)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for fragment in (
        "def evaluate", "def predict", "torch.inference_mode",
        "torch.random.fork_rng", "default_generator.manual_seed",
        "zero_grad", "finally:", "for batch in", "return",
    ):
        assert fragment in code, fragment
    assert "lispyn.workflow" not in code
    assert all(cell.execution_count is None and not cell.outputs
               for cell in notebook.cells if cell.cell_type == "code")


def test_composition_runs_on_cpu_and_restores_modes_rng_and_gradients(tmp_path):
    pytest.importorskip("torch")
    executable = shutil.which("hy")
    assert executable, "Hy CLI expected in the test environment"
    source = tmp_path / "macro_composition.hy"
    source.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run([executable, str(source)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "macro-composition-passed" in result.stdout


@pytest.mark.parametrize("form", [
    '(defeval-step "invalid" model criterion)',
    '(defpredict-step "invalid" model)',
    "(collect-batches [batch] loader (print batch))",
    "(collect-batches batch loader)",
    "(with-torch-seed 12)",
    "(with-clean-gradients optimizer)",
])
def test_new_macro_invalid_syntax_is_rejected(tmp_path, form):
    source = tmp_path / "bad.hy"
    source.write_text(
        "(require lispyn.macros [defeval-step defpredict-step collect-batches "
        "with-torch-seed with-clean-gradients])\n" + form + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "lispyn.cli", "build", str(source)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, form


def test_catalog_and_agent_instructions_refer_to_all_new_macros():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    catalog = (ROOT / "supported_macros" / "README.md").read_text(encoding="utf-8")
    agent_guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    macros = ("defeval-step", "defpredict-step", "collect-batches",
              "with-torch-seed", "with-clean-gradients")
    for macro in macros:
        assert "### `" + macro + "`" in catalog
        assert macro in readme
    assert "For humans and AI agents" in readme
    assert "compile time" in agent_guide
