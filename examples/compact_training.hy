; %% [markdown]
; ## One-call training with metrics and best checkpoint
; A tiny offline classification experiment. No GPU or downloads.
; %% [code]
(import torch)
(import lispyn.training [fit load-checkpoint])

(torch.manual_seed 42)
(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv dataset (torch.utils.data.TensorDataset x y))
(setv batches (torch.utils.data.DataLoader dataset :batch_size 2))
(setv model (torch.nn.Linear 2 2))
(setv optimizer (torch.optim.Adam (.parameters model) :lr 0.05))

(setv history
  (fit model batches optimizer
    :loss-fn (torch.nn.CrossEntropyLoss)
    :validation batches
    :epochs 3
    :metrics "accuracy"
    :save-best "artifacts/best.pt"
    :log-path "artifacts/history.json"))

(print "validation accuracy:" (get history "val_accuracy"))
(print "best validation loss:" (min (get history "val_loss")))
; %% [markdown]
; ## Checkpoint
; The best checkpoint is on disk; the in-memory model stays at the last epoch.
; %% [code]
(load-checkpoint model "artifacts/best.pt")
(print "restored best checkpoint")
