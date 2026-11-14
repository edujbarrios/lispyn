"""Create a minimal, immediately useful Lispyn notebook experiment.

Copyright 2026 Eduardo J. Barrios. SPDX-License-Identifier: Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path

from .compiler import LispynError


STARTER = """; Lispyn experiment — https://github.com/edujbarrios/lispyn
; Original toolkit by Eduardo J. Barrios.
; %% [markdown]
; # Your reproducible experiment
; Declare parameters and record why you chose them.
; %% [code]
(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 3)
  (setv learning-rate 0.01))
; %% [markdown]
; ## Train a small PyTorch baseline
; Synthetic data makes this template runnable without downloads.
; %% [code]
(import torch)
(import lispyn.workflow [train])
(require lispyn.macros [with-seed])
(require lispyn.macros [report-run])

(with-seed seed
  (setv model (torch.nn.Linear 2 2)))

(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv data (torch.utils.data.TensorDataset x y))
(setv result
  (train model data
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs epochs
    :seed seed
    :batch-size 2
    :lr learning-rate
    :device "cpu"))
(report-run result
  :title "Baseline experiment"
  :notes "Synthetic data — replace with a real task and held-out validation."
  :save "experiment_report.md")
; %% [markdown]
; ## Next steps
; Add independent validation data, real-world context, and a conclusion.
; Keep sensitive data out of saved reports and Markdown cells.
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
