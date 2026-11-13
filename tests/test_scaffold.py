"""The project starter must be safe to create and valid as Hy source."""

import nbformat

from lispyn.cli import main
from lispyn.notebooks import build_notebook


def test_new_file_is_created_and_has_notebook_markers(tmp_path):
    file = tmp_path / "nested" / "experiment.hy"
    assert main(["new", str(file)]) == 0
    content = file.read_text()
    assert "(require lispyn.macros [with-seed])" in content
    assert "(import lispyn.workflow [train])" in content
    assert "Eduardo J. Barrios" in content
    assert "; %% [markdown]" in content
    assert "; %% [code]" in content


def test_new_never_overwrites_existing_sources(tmp_path, capsys):
    file = tmp_path / "original.hy"
    file.write_text("(print 123)\n")
    assert main(["new", str(file)]) == 1
    assert file.read_text() == "(print 123)\n"
    assert "not overwriting" in capsys.readouterr().err


def test_new_rejects_wrong_extension(tmp_path, capsys):
    file = tmp_path / "experiment.py"
    assert main(["new", str(file)]) == 1
    assert not file.exists()
    assert ".hy filename" in capsys.readouterr().err
