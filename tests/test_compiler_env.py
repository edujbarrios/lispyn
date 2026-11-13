"""Hy executable resolution must work without activating a venv."""

import sys
from pathlib import Path
from types import SimpleNamespace

from lispyn import compiler


def test_hy_scripts_found_next_to_python_even_when_not_in_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.touch()
    hy2py = bin_dir / "hy2py"
    hy2py.touch()
    hy = bin_dir / "hy"
    hy.touch()
    monkeypatch.setattr(compiler.sys, "executable", str(python))
    monkeypatch.setattr(compiler.shutil, "which", lambda name: None)
    assert compiler.find_hy_executable("hy2py") == str(hy2py)
    assert compiler.find_hy_executable("hy") == str(hy)


def test_hy_script_falls_back_to_path(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(compiler.sys, "executable", str(python))
    monkeypatch.setattr(compiler.shutil, "which",
                        lambda name: "/somewhere/" + name)
    assert compiler.find_hy_executable("hy2py") == "/somewhere/hy2py"


def test_run_hy_uses_current_environment_script(tmp_path, monkeypatch):
    source = tmp_path / "experiment.hy"
    source.write_text("(print 3)\n")
    exe = tmp_path / "hy"
    exe.touch()
    monkeypatch.setattr(compiler, "find_hy_executable", lambda name: str(exe))
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    assert compiler.run_hy(source) == 0
    assert commands[0][0] == [str(exe), str(source)]
