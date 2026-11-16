"""Keep the public hello demo and its three commands accurate."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_readme_hello_snippet_is_the_runnable_source():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "examples/hello.hy").read_text(encoding="utf-8")
    match = re.search(
        r"## Quick start: the same `hello.hy` in three modes.*?```hy\n(.*?)\n```",
        readme, re.S,
    )
    assert match is not None
    # Comments explain internals to future maintainers; the README shows
    # exactly the executable Hy forms that users can copy.
    expected = "\n".join(
        line for line in source.splitlines() if not line.startswith(";")
    ).strip()
    assert match.group(1).strip() == expected


def test_readme_explains_three_distinct_execution_artifacts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Why Lispyn for research?" in readme
    assert "git clone https://github.com/edujbarrios/lispyn.git" in readme
    assert "`lispyn run examples/hello.hy`" not in readme  # show copyable shell blocks
    assert "lispyn run examples/hello.hy\n" in readme
    assert "lispyn build examples/hello.hy -o hello.ipynb\n" in readme
    assert "lispyn execute examples/hello.hy -o hello-executed.ipynb\n" in readme
    assert "Nothing is executed" in readme
    assert "saves the executed cells" in readme
    assert "PyTorch is installed separately" in readme
