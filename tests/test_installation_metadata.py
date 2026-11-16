"""Installation defaults and optional research environment stay consistent."""

from importlib import metadata, util
from pathlib import Path

from packaging.requirements import Requirement

from lispyn import __version__
from lispyn.notebooks import build_notebook


ROOT = Path(__file__).resolve().parents[1]


def test_plain_lispyn_installs_notebook_and_plot_stack():
    """These must be unconditional distribution requirements, not extras."""
    deps = [Requirement(item) for item in (metadata.requires("lispyn") or [])]
    base = {dep.name.lower() for dep in deps if dep.marker is None}
    assert {"hy", "nbformat", "packaging", "nbclient",
            "ipykernel", "matplotlib"} <= base
    assert "torch" not in base, "PyTorch must stay platform-selectable"
    assert metadata.version("lispyn") == __version__
    for module in ("hy", "nbformat", "nbclient", "ipykernel", "matplotlib"):
        assert util.find_spec(module) is not None, module


def test_requirements_txt_is_a_documented_optional_research_environment():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirements = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements.count("-e .") == 1
    packages = {
        Requirement(line).name.lower()
        for line in requirements if not line.startswith("-e ")
    }
    assert {"torch", "numpy", "pandas", "matplotlib",
            "nbclient", "ipykernel"} <= packages
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install lispyn" in readme
    assert "pip install -r requirements.txt" in readme
    assert "PyTorch is installed separately" in readme


def test_pip_install_macro_creates_a_cell_but_never_installs_on_build(tmp_path):
    source = tmp_path / "optional_dependency.hy"
    source.write_text(
        '(require lispyn.macros [pip-install])\n'
        '(pip-install "scikit-learn>=1.5")\n'
        '(print "not executed")\n',
        encoding="utf-8",
    )
    notebook = build_notebook(source)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    tagged = [
        cell for cell in cells
        if "lispyn-pip-install" in cell.metadata.get("tags", [])
    ]
    assert len(tagged) == 1
    assert "scikit-learn>=1.5" in tagged[0].source
    assert all(cell.get("execution_count") is None
               and not cell.get("outputs") for cell in cells)
    assert "not executed" in cells[-1].source
