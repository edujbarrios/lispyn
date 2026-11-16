"""The landing README stays short and distinguishes each Hy execution mode."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_to_real_example_and_complete_catalog():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "examples" / "hello.hy").read_text(encoding="utf-8")
    catalog = (ROOT / "supported_macros" / "README.md").read_text(encoding="utf-8")
    assert "examples/hello.hy" in readme
    assert "supported_macros/README.md" in readme
    assert "All 49 macros" in readme and "49 macros" in catalog
    assert '(print "Lispyn checks passed!")' in source
    assert len(readme) < 3400


def test_readme_explains_three_distinct_artifacts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Why use it for research?" in readme
    assert "git clone https://github.com/edujbarrios/lispyn.git" in readme
    assert "lispyn run examples/hello.hy\n" in readme
    assert "lispyn build examples/hello.hy -o hello.ipynb\n" in readme
    assert "lispyn execute examples/hello.hy -o hello-executed.ipynb\n" in readme
    assert "Nothing is executed" in readme
    assert "saves the executed cells" in readme
    assert "PyTorch is installed separately" in readme
