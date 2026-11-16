; Minimal, REAL PyTorch training demo. Data are small and synthetic.
; The held-out test subset is intentionally not used for tuning.
(require lispyn.macros [md])
(md "# A reproducible PyTorch experiment"
    "A tiny, synthetic classification baseline with a held-out test split."
    "Lispyn turns Hy source into executable Python/Jupyter cells.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 6))

; Each generated notebook cell compiles independently: require macros again.
(require lispyn.macros [train-val-test train-until report-run narrate])
(import torch)

(torch.manual_seed seed)
(setv features
  (torch.tensor
    [[0.0 0.0] [0.0 1.0] [1.0 0.0] [1.0 1.0]
     [0.2 0.3] [0.3 0.9] [0.9 0.3] [0.8 0.9]]))
(setv labels (torch.tensor [0 1 0 1 0 1 0 1]))
(setv dataset (torch.utils.data.TensorDataset features labels))
(setv [training validation held-out]
  (train-val-test dataset
    :validation-fraction 0.25
    :test-fraction 0.25
    :seed seed))

(setv model (torch.nn.Linear 2 2))
(setv result
  (train-until model training 2
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs epochs :batch-size 2 :lr 0.05 :seed seed :device "cpu"))

(report-run result :title "Synthetic classification baseline")
(narrate "Best validation loss: **{loss:.3f}**"
  :loss (min (get result.history "val_loss")))
(print "Held-out test rows reserved for final evaluation:" (len held-out))
