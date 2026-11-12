"""Sectioned notebooks preserve source-level Hy comments and cell order."""

from unittest.mock import patch

import pytest

from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook, split_sections


def test_split_hy_comments_into_markdown_and_code():
    source = """(print "start")
; %% [markdown]
; ## Process dataset
; CSVs are handled by pandas.
;
; Data is reused below.
; %% [code]
(print "end")
"""
    assert split_sections(source) == [
        ("code", '(print "start")'),
        ("markdown", "## Process dataset\nCSVs are handled by pandas.\n\nData is reused below."),
        ("code", '(print "end")'),
    ]


def test_markdown_section_requires_hy_comments():
    with pytest.raises(LispynError, match="comment lines"):
        split_sections("; %% [markdown]\n# Not a Hy comment\n")


def test_build_separate_code_cells_with_source_order(tmp_path):
    src = tmp_path / "dataset.hy"
    src.write_text(
        '(setv x 10)\n; %% [markdown]\n; ## Results\n; %% [code]\n(print x)\n'
    )
    with patch("lispyn.notebooks.compile_section", side_effect=["x = 10", "print(x)"]) as mock:
        notebook = build_notebook(src)
    assert [c.cell_type for c in notebook.cells] == ["markdown", "code", "markdown", "code"]
    assert [c.source for c in notebook.cells[1:]] == ["x = 10", "## Results", "print(x)"]
    assert mock.call_count == 2
