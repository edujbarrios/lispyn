"""Keep the public Lisp macro registry synchronized with its Hy implementation.

The catalog is the only macro reference; every defmacro must be documented
exactly once and link to a source/usage example. This deliberately tests the
repository files, not a hand-maintained list of names.
"""
from pathlib import Path
import re

import nbformat

from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]
MACROS = ROOT / "src/lispyn/macros.hy"
CATALOG = ROOT / "supported_macros/README.md"


def test_every_hy_macro_is_documented_once():
    source = MACROS.read_text(encoding="utf-8")
    documentation = CATALOG.read_text(encoding="utf-8")
    defined = re.findall(r"(?m)^\(defmacro\s+([a-z][\w-]*)\b", source)
    entries = re.findall(r"(?m)^### `([a-z][\w-]*)`\s*$", documentation)
    assert len(defined) == len(set(defined)), "duplicate public Hy macro definitions"
    assert len(entries) == len(set(entries)), "duplicate macro catalog entries"
    assert set(defined) == set(entries), (
        f"Macros missing from catalog: {sorted(set(defined) - set(entries))}; "
        f"undocumented catalog entries: {sorted(set(entries) - set(defined))}"
    )
    assert len(defined) >= 84, "The supported API must not silently shrink"
    for name in defined:
        section = re.search(
            rf"(?ms)^### `{re.escape(name)}`\s*\n(.+?)(?=^### |^## |\Z)",
            documentation,
        )
        assert section, name
        assert "**Purpose:**" in section.group(1), f"{name} lacks an explanation"
        assert "hy\n" in section.group(1), f"{name} lacks a Hy example"
        assert name in section.group(1), f"{name} not used in its example"


def test_root_readme_is_ai_notebook_first_and_links_to_macro_registry():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "supported_macros/README.md" in readme
    assert "examples/quickstart.hy" in readme
    assert "pip install lispyn" in readme
    assert "pip install -r requirements.txt" in readme
    assert "lispyn execute examples/hello.hy" in readme
    assert "Lisp-powered AI notebooks" in readme
    assert "Why use it for research?" not in readme
    assert len(readme) < 7500
    assert "MACRO_AUTHORING.md" in readme
    assert all(name in readme for name in ("PyTorch", "Matplotlib", "pandas", "NumPy"))
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "blob/main/supported_macros/README.md" in project
    assert not (ROOT / "docs").exists(), "obsolete docs/ must be fully removed"


def test_quickstart_is_a_real_compilable_hy_experiment():
    nb = build_notebook(ROOT / "examples/quickstart.hy")
    nbformat.validate(nb)
    assert sum(cell.cell_type == "markdown" for cell in nb.cells) >= 2
    assert any("parameters" in cell.metadata.get("tags", [])
               for cell in nb.cells if cell.cell_type == "code")
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "split_dataset_three" in code
    assert "lispyn.workflow" in code
    assert "show_run_report" in code
    assert all(cell.get("execution_count") is None
               for cell in nb.cells if cell.cell_type == "code")
