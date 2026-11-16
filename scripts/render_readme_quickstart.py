"""Generate README's cell-by-cell demo from a real executed Hy notebook.

Read actual saved Jupyter source, stdout, Markdown and the saved Matplotlib
figure. Never train, invent results, or replace failed/incomplete evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/quickstart.hy"
NOTEBOOK = ROOT / "example_generated_notebooks/quickstart_executed.ipynb"
FIGURE = ROOT / "example_generated_notebooks/assets/quickstart_training_curves.png"
README = ROOT / "README.md"
BEGIN = "<!-- QUICKSTART_EXECUTED:START -->"
END = "<!-- QUICKSTART_EXECUTED:END -->"
TICK = chr(96)


def _value(v):
    return "".join(v) if isinstance(v, list) else str(v)


def _fence(value, language):
    value = _value(value).rstrip("\n")
    n = max((len(x.group()) + 1 for x in re.finditer(TICK + "+", value)), default=3)
    bar = TICK * max(3, n)
    return f"{bar}{language}\n{value}\n{bar}"


def render(source=SOURCE, notebook_path=NOTEBOOK, figure=FIGURE):
    source, notebook_path, figure = map(Path, (source, notebook_path, figure))
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    digest = sha256(source.read_bytes()).hexdigest()
    prov = nb.metadata.get("lispyn", {})
    if prov.get("source") != source.name or prov.get("source_sha256") != digest:
        raise ValueError("Notebook provenance does not match original Hy source")
    if (not figure.is_file() or figure.stat().st_size < 3000
            or not figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Training figure was not generated as a valid PNG")
    executable = [c for c in nb.cells if c.cell_type == "code"]
    if not executable or any(c.execution_count is None for c in executable):
        raise ValueError("Notebook is not completely executed")
    if any(o.output_type == "error" for c in executable for o in c.outputs):
        raise ValueError("Notebook contains a recorded execution error")
    if "plot_history" not in executable[-1].source:
        raise ValueError("The final code cell must plot the recorded training history")

    code_link = f"{TICK}examples/quickstart.hy{TICK}"
    ipynb_link = f"{TICK}quickstart_executed.ipynb{TICK}"
    lines = [
        "### The original Hy experiment", "",
        f"Exactly the source in [{code_link}](examples/quickstart.hy)."
        " The classifier really trains in PyTorch on tiny synthetic data, with"
        " an independent held-out test subset.", "",
        _fence(source.read_text(encoding="utf-8"), "hy"), "",
        "### Every executed Jupyter cell, with actual outputs", "",
        f"Captured from [{ipynb_link}](example_generated_notebooks/quickstart_executed.ipynb)."
        " Click each cell to inspect the generated Python. All outputs shown"
        " below are copied from the saved executed notebook.", "",
    ]

    image_seen = False
    for index, cell in enumerate(nb.cells, 1):
        if cell.cell_type == "markdown":
            lines += [
                f"**Cell {index} · Markdown**", "",
                "<details><summary>View notebook Markdown source</summary>", "",
                _fence(cell.source, "markdown"), "",
                "</details>", "",
            ]
            continue
        if cell.cell_type == "raw":
            lines += [f"**Cell {index} · Raw cell**", "", _fence(cell.source, "text"), ""]
            continue
        if cell.cell_type != "code":
            raise ValueError(f"Unsupported cell type: {cell.cell_type}")
        lines += [
            f"**Cell {index} · Code · executed #{cell.execution_count}**", "",
            "<details><summary>View generated Python</summary>", "",
            _fence(cell.source, "python"), "",
            "</details>", "",
        ]
        if not cell.outputs:
            lines += ["_No visible output (imports/setup)._ ", ""]
        for output in cell.outputs:
            if output.output_type == "stream":
                value = _value(output.get("text", ""))
                if value.strip():
                    lines += ["**Recorded stdout/stderr:**", "", _fence(value, "text"), ""]
            elif output.output_type in ("display_data", "execute_result"):
                data = output.get("data", {})
                if "text/markdown" in data:
                    lines += ["**Rendered Markdown:**", "",
                              _value(data["text/markdown"]).strip(), ""]
                elif "image/png" in data:
                    image_seen = True
                    lines += [
                        "**Final chart — train vs. validation loss from this execution:**", "",
                        "![Real training history from the executed Lispyn Hy quickstart]"
                        "(example_generated_notebooks/assets/quickstart_training_curves.png)", "",
                    ]
                elif "text/plain" in data:
                    lines += ["**Recorded output:**", "",
                              _fence(data["text/plain"], "text"), ""]
                elif "text/html" in data:
                    lines += ["_HTML display omitted from README._", ""]
            else:
                raise ValueError(f"Unexpected Jupyter output: {output.output_type}")

    if not image_seen:
        lines += [
            "**Final chart — saved from the last executed cell:**", "",
            "![Real training history from the executed Lispyn Hy quickstart]"
            "(example_generated_notebooks/assets/quickstart_training_curves.png)", "",
        ]

    lines += [
        f"**Source SHA-256:** {TICK}{digest}{TICK} (recorded in the executed notebook).", "",
        "You can verify it yourself:", "",
        _fence("lispyn audit example_generated_notebooks/quickstart_executed.ipynb "
               "--source examples/quickstart.hy", "bash"), "",
        "*These are measured outputs from a toy synthetic dataset,"
        " not research-quality benchmarks.*", "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def rewrite(readme_path, section, *, check=False):
    readme_path = Path(readme_path)
    old = readme_path.read_text(encoding="utf-8")
    if old.count(BEGIN) != 1 or old.count(END) != 1:
        raise ValueError("Expected exactly one pair of README demo markers")
    updated = (old.split(BEGIN, 1)[0] + BEGIN + "\n"
               + section.rstrip() + "\n" + END
               + old.split(END, 1)[1])
    if check and old != updated:
        raise ValueError("README notebook output snapshot does not match the executed notebook")
    if not check:
        readme_path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    section = render()
    rewrite(README, section, check=args.check)
    print(f"{'Verified' if args.check else 'Generated'} README from "
          f"{NOTEBOOK.name}: {len(section)} characters")


if __name__ == "__main__":
    main()
