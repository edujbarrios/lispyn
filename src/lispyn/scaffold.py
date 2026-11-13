"""Create a minimal, immediately useful Lispyn notebook experiment.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path

from .compiler import LispynError


STARTER = """; Lispyn experiment — https://github.com/edujbarrios/lispyn
; Original toolkit by Eduardo J. Barrios.
; %% [markdown]
; ## First experiment
; Write Lisp here; Lispyn generates Python Jupyter notebook cells.
; %% [code]
(import torch)
(import lispyn.workflow [train])
(require lispyn.macros [with-seed])

(with-seed 42
  (setv model (torch.nn.Linear 2 2)))

(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv data (torch.utils.data.TensorDataset x y))
(setv result
  (train model data
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 3
    :batch-size 2
    :lr 0.01
    :device "cpu"))
(print "training loss:" (get result.history "train_loss"))
; %% [markdown]
; ## Next steps
; Replace the toy data with your own images, text, audio or CSV dataset.
; Add validation data and checkpoints when you're ready.
"""


def create_experiment(path: str | Path) -> Path:
    """Create a new source file without overwriting existing work.

    The file is not run and no notebook is generated automatically.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix != ".hy":
        raise LispynError("New Lispyn experiments must use a .hy filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(STARTER)
    except FileExistsError as exc:
        raise LispynError(f"File already exists; not overwriting: {path}") from exc
    return path
