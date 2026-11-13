; %% [markdown]
; ## A reproducible Lispyn run
; This dataset is intentionally tiny. In a real project, use separate validation data.
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
    :epochs 2
    :batch-size 2
    :validation data
    :metrics "accuracy"
    :seed 42
    :device "cpu"
    :experiment-dir "runs/smoke"))
(print "metrics:" result.history)
