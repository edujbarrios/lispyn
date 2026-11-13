"""Hy macro smoke tests: compile-time expansion and runtime behavior."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_with_eval_restores_train_mode_after_exception(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("hy")
    from lispyn.notebooks import build_notebook

    source = tmp_path / "infer.hy"
    source.write_text(
        '(require lispyn.macros [with-eval])\n'
        '(import torch)\n'
        '(setv model (torch.nn.Linear 2 2))\n'
        '(try\n'
        '  (with-eval model\n'
        '    (assert (not model.training))\n'
        '    (assert (not (torch.is-grad-enabled)))\n'
        '    (raise (RuntimeError "test")))\n'
        '  (except [RuntimeError] (print "caught")))\n'
        '(assert model.training)\n'
        '(print "restored")\n',
        encoding="utf-8",
    )
    executable = shutil.which("hy")
    assert executable, "Hy executable missing despite installed package"
    result = subprocess.run([executable, str(source)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "caught" in result.stdout
    assert "restored" in result.stdout
    notebook = build_notebook(source)
    assert "inference_mode" in notebook.cells[-1].source
