; %% [markdown]
; ## One-call PyTorch training
; No downloads or GPU: native tensors, native model, Lispyn orchestration.
; %% [code]
(import torch)
(import lispyn.workflow [train])

(torch.manual_seed 42)
(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv data (torch.utils.data.TensorDataset x y))
(setv model (torch.nn.Linear 2 2))
(setv run
  (train model data
    :loss-fn (torch.nn.CrossEntropyLoss)
    :validation data
    :epochs 2
    :batch-size 2
    :lr 0.05
    :metrics "accuracy"
    :device "cpu"
    :save-best "artifacts/one-call-best.pt"))
(print "device:" run.device)
(print "validation accuracy:" (get run.history "val_accuracy"))
