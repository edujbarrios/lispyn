"""Render a short, honest README demo from an actually executed notebook.

The visible Hy snippet is extracted verbatim from the runnable quickstart.
The result lines and figure come from the executed .ipynb. No experiment code
runs during rendering, and stale/failed notebooks are rejected.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/quickstart.hy"
NOTEBOOK = ROOT / "example_generated_notebooks/quickstart_executed.ipynb"
FIGURE = ROOT / "example_generated_notebooks/assets/quickstart_training_curves.png"
README = ROOT / "README.md"
BEGIN = "<!-- QUICKSTART_EXECUTED:START -->"
END = "<!-- QUICKSTART_EXECUTED:END -->"


def _text(value):
    return "".join(value) if isinstance(value, list) else str(value)


def _single(values, name):
    if len(values) != 1:
        raise ValueError(f"Expected one recorded {name}, found {len(values)}")
    return values[0]


def _hy_snippet(source):
    """Use exact, executable slices of the real example rather than fake Hy."""
    first = "(require lispyn.macros [train-until])"
    next_cell = "(require lispyn.macros [md])"
    last = "(require lispyn.macros [curve])"
    if first not in source or last not in source:
        raise ValueError("Quickstart no longer defines train-until and curve")
    start = source.index(first)
    stop = source.find(next_cell, start)
    if stop < start:
        raise ValueError("Training cell could not be isolated")
    training = source[start:stop].strip()
    plotting = source[source.rindex(last):].strip()
    return training + "\n\n" + plotting


def render(source=SOURCE, notebook_path=NOTEBOOK, figure=FIGURE):
    source, notebook_path, figure = map(Path, (source, notebook_path, figure))
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    fingerprint = sha256(source.read_bytes()).hexdigest()
    meta = notebook.metadata.get("lispyn", {})
    if meta.get("source") != source.name or meta.get("source_sha256") != fingerprint:
        raise ValueError("Notebook provenance does not match Hy source")
    if (not figure.is_file() or figure.stat().st_size < 3000
            or not figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Real training figure is missing")
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code or any(cell.get("execution_count") is None for cell in code):
        raise ValueError("All notebook code cells must be executed")
    if any(item.output_type == "error" for cell in code for item in cell.outputs):
        raise ValueError("Executed notebook contains an execution error")
    if "plot_history" not in code[-1].source or not any(
        "image/png" in item.get("data", {}) for item in code[-1].outputs
    ):
        raise ValueError("Final cell must display the recorded training plot")
    stdout, markdown = [], []
    for cell in code:
        for item in cell.outputs:
            if item.output_type == "stream":
                stdout.extend(_text(item.get("text", "")).splitlines())
            elif item.output_type in ("execute_result", "display_data"):
                if "text/markdown" in item.get("data", {}):
                    markdown.append(_text(item.data["text/markdown"]).strip())
    split = _single([line for line in stdout if line.startswith("Samples:")], "dataset split")
    epochs = _single([line for line in stdout if line.startswith("Completed epochs:")],
                     "epoch count")
    accuracy = _single(
        [text for text in markdown if text.startswith("Held-out test accuracy:")],
        "held-out accuracy",
    )
    _single([text for text in markdown if text.startswith("### Recorded experiment scores")],
            "metric table")
    _single([text for text in markdown
             if text.startswith("# PyTorch baseline · recorded metrics")],
            "training report")
    snippet = _hy_snippet(source.read_text(encoding="utf-8"))
    return "\n".join([
        "**Hy syntax** — an excerpt from the [full runnable experiment](examples/quickstart.hy)"
        " (native PyTorch; tiny synthetic dataset):", "",
        "§§§hy", snippet, "§§§", "",
        "**Real recorded output:**", "",
        f"§{split}§ · §{epochs}§", "",
        accuracy, "",
        "![Actual training and validation loss from the Hy quickstart]"
        "(example_generated_notebooks/assets/quickstart_training_curves.png)", "",
        "[View every executed Jupyter cell]"
        "(example_generated_notebooks/quickstart_executed.ipynb)"
        " · [Full Hy source](examples/quickstart.hy)", "",
        "_Toy data for workflow demonstration only; not a scientific benchmark._", "",
    ]).replace("§", chr(96)).rstrip() + "\n"


def rewrite(readme_path, section, *, check=False):
    readme_path = Path(readme_path)
    previous = readme_path.read_text(encoding="utf-8")
    if previous.count(BEGIN) != 1 or previous.count(END) != 1:
        raise ValueError("README needs exactly one pair of quickstart markers")
    updated = (previous.split(BEGIN, 1)[0] + BEGIN + "\n"
               + section.rstrip() + "\n" + END
               + previous.split(END, 1)[1])
    if check and previous != updated:
        raise ValueError("README snapshot is stale: regenerate it from executed notebook")
    if not check:
        readme_path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    # The landing README stays brief: verify real notebook provenance
    # without injecting experimental results into the README.
    if not text:
        raise ValueError("Empty executed quickstart evidence")
    print("Verified executed notebook provenance and training plot")


if __name__ == "__main__":
    main()
