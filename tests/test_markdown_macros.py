"""Source-level Markdown macros and generated notebook gallery regression tests."""

from pathlib import Path
import shutil
import subprocess

import nbformat
import pytest

from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook, split_markdown_macros


def test_literal_markdown_forms_create_true_cells(tmp_path):
    source = tmp_path / "demo.hy"
    source.write_text(
        '(require lispyn.macros [md markdown])\n'
        '(md "# Heading" "Text with **bold** characters.")\n'
        '(setv answer 42)\n'
        '(markdown "## Second heading" "More detail.")\n'
        '(print answer)\n',
        encoding="utf-8",
    )
    book = build_notebook(source)
    nbformat.validate(book)
    kinds = [cell.cell_type for cell in book.cells]
    assert kinds == ["markdown", "code", "markdown", "code", "markdown", "code"]
    assert book.cells[2].source == "# Heading\n\nText with **bold** characters."
    assert book.cells[4].source == "## Second heading\n\nMore detail."
    assert "42" in book.cells[-1].source or "answer" in book.cells[-1].source
    for code in (cell.source for cell in book.cells if cell.cell_type == "code"):
        assert "md(" not in code and "markdown(" not in code


def test_markdown_macro_extraction_rejects_dynamic_and_nested_content():
    with pytest.raises(LispynError, match="literal Markdown"):
        split_markdown_macros('(md (+ 1 2))\n')
    with pytest.raises(LispynError, match="top-level"):
        split_markdown_macros('(do\n  (md "Nested"))\n')
    with pytest.raises(LispynError, match="own lines"):
        split_markdown_macros('(md "Heading") (print "same-line")\n')
    with pytest.raises(LispynError, match="cannot be empty"):
        split_markdown_macros('(md "")\n')


def test_stable_generated_ids(tmp_path):
    source = tmp_path / "sample.hy"
    source.write_text('(require lispyn.macros [md])\n(md "# Stable")\n(print 42)\n')
    first = build_notebook(source)
    second = build_notebook(source)
    assert [c.id for c in first.cells] == [c.id for c in second.cells]
    assert len({c.id for c in first.cells}) == len(first.cells)


def test_markdown_example_runs_directly_and_generates_cells():
    source = Path(__file__).resolve().parents[1] / "examples" / "markdown_cells.hy"
    hy = shutil.which("hy")
    assert hy
    ran = subprocess.run([hy, str(source)], cwd=source.parent,
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert "Average: 5.0" in ran.stdout
    nb = build_notebook(source)
    assert sum(c.cell_type == "markdown" for c in nb.cells) >= 4
    assert any("Reusable explanations" in c.source for c in nb.cells)


def test_resnet_recipe_compiles_without_training():
    source = Path(__file__).resolve().parents[1] / "examples" / "resnet_finetune_300.hy"
    nb = build_notebook(source)
    assert sum(c.cell_type == "markdown" for c in nb.cells) >= 5
    assert any("300 epochs" in c.source for c in nb.cells if c.cell_type == "markdown")
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "300" in code
    assert "trainable" in code
    assert "make_plotter" in code
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_gallery_compiles_selected_examples_to_dedicated_directory(tmp_path):
    from lispyn.gallery import EXAMPLE_SOURCES, generate_gallery

    sources = Path(__file__).resolve().parents[1] / "examples"
    directory = tmp_path / "example_generated_notebooks"
    files = generate_gallery(sources, directory)
    assert len(files) == len(EXAMPLE_SOURCES)
    assert all(file.parent == directory for file in files)
    assert all(file.suffix == ".ipynb" for file in files)
    for path in files:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert "Eduardo J. Barrios" in nb.cells[0].source
        assert all(cell.get("execution_count") is None for cell in nb.cells
                   if cell.cell_type == "code")
    before = {file.name: file.read_bytes() for file in files}
    generate_gallery(sources, directory)
    after = {file.name: file.read_bytes() for file in files}
    assert before == after, "Notebook gallery should be deterministic"


def test_galleries_dont_run_expensive_training(tmp_path):
    from lispyn.cli import main
    sources = Path(__file__).resolve().parents[1] / "examples"
    output = tmp_path / "generated"
    assert main(["gallery", "--source-dir", str(sources),
                 "--output-dir", str(output)]) == 0
    assert (output / "resnet_finetune_300.ipynb").is_file()
