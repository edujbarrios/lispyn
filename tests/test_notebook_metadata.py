"""Source-level raw cells, cell tags and compiled Python notebook output."""

from pathlib import Path
import shutil
import subprocess

import nbformat
import pytest

from lispyn.compiler import LispynError
from lispyn.notebooks import build_notebook, split_markdown_macros


def test_raw_and_tagged_code_cells_compiled_and_valid(tmp_path):
    source = tmp_path/'metadata.hy'
    source.write_text(
        '(require lispyn.macros [raw-cell cell-tags md])\n'
        '(raw-cell "<div>Raw HTML or TeX for another exporter</div>")\n'
        '(require lispyn.macros [cell-tags])\n'
        '(cell-tags "parameters" "hide-input")\n'
        '(setv epochs 300)\n'
        '(md "## Results")\n'
        '(require lispyn.macros [cell-tags])\n'
        '(cell-tags "lispyn-pip-install" "hide-input")\n'
        '(pip-install "math")\n'
    )
    nb = build_notebook(source)
    nbformat.validate(nb)
    assert [c.cell_type for c in nb.cells] == [
        'markdown', 'code', 'raw', 'code', 'code', 'markdown', 'code', 'code'
    ]
    assert nb.cells[2].source.startswith('<div>Raw')
    assert nb.cells[4].metadata['tags'] == ['parameters', 'hide-input']
    assert 'epochs' in nb.cells[4].source
    assert nb.cells[-1].metadata['tags'] == ['lispyn-pip-install', 'hide-input']
    assert all(c.get('execution_count') is None for c in nb.cells if c.cell_type=='code')


def test_stable_ids_with_cell_metadata(tmp_path):
    source = tmp_path/'a.hy'
    source.write_text('(cell-tags "parameters")\n(setv a 1)\n')
    first, second = build_notebook(source), build_notebook(source)
    assert [c.id for c in first.cells] == [c.id for c in second.cells]


@pytest.mark.parametrize('source, error', [
    ('(raw-cell (+ 1 2))\n', 'literal strings'),
    ('(raw-cell "")\n', 'cannot be empty'),
    ('(cell-tags)\n', 'literal tag strings'),
    ('(cell-tags "not a tag")\n', 'tag names'),
    ('(cell-tags "a" "a")\n', 'unique'),
    ('(cell-tags "parameters")\n', 'followed by a code cell'),
    ('(cell-tags "parameters")\n(md "header")\n', 'followed by a code cell'),
    ('(cell-tags "parameters")\n(cell-tags "hide-input")\n(print 2)\n', 'exactly one code cell'),
    ('(raw-cell "x") (print 3)\n', 'own lines'),
    ('(do\n (raw-cell "x"))\n', 'top-level'),
])
def test_invalid_directives(source, error, tmp_path):
    file = tmp_path/'invalid.hy'
    file.write_text(source)
    with pytest.raises(LispynError, match=error):
        build_notebook(file)


def test_hy_raw_and_tag_macros_are_direct_run_no_ops(tmp_path):
    src = tmp_path/'example.hy'
    src.write_text(
        '(require lispyn.macros [raw-cell cell-tags])\n'
        '(raw-cell "<h1>title</h1>")\n'
        '(cell-tags "parameters")\n'
        '(setv epochs 5)\n'
        '(print "tags-ok" epochs)\n'
    )
    hy = shutil.which('hy')
    assert hy
    run = subprocess.run([hy, str(src)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'tags-ok 5' in run.stdout
    nb = build_notebook(src)
    nbformat.validate(nb)
