"""Small CPU smoke test for reusable head-only and full fine-tuning macros."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_head_and_full_tuning_macros(tmp_path):
    pytest.importorskip("torch")
    hy = shutil.which("hy")
    assert hy
    source = tmp_path / "phases.hy"
    source.write_text(
        '(require lispyn.macros [head-tune full-tune])\n'
        '(import torch)\n'
        '(setv data (torch.utils.data.TensorDataset '
        '(torch.randn 6 2) (torch.tensor [0 1 0 1 0 1])))\n'
        '(setv model (torch.nn.Sequential '
        '(torch.nn.Linear 2 4) (torch.nn.ReLU) (torch.nn.Linear 4 2)))\n'
        '(setv first (head-tune model data "2" :epochs 1 :batch-size 2 '
        ':loss-fn (torch.nn.CrossEntropyLoss) :device "cpu"))\n'
        '(assert (not (getattr (getattr (get model 0) "weight") "requires_grad")))\n'
        '(setv second (full-tune model data :epochs 1 :batch-size 2 '
        ':loss-fn (torch.nn.CrossEntropyLoss) :device "cpu"))\n'
        '(assert (getattr (getattr (get model 0) "weight") "requires_grad"))\n'
        '(assert (= (len (get second.history "train_loss")) 1))\n'
        '(print "two-stages-ok")\n'
    )
    r = subprocess.run([hy, str(source)], cwd=tmp_path,
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert "two-stages-ok" in r.stdout
