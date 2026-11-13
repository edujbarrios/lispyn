; Early stopping with native PyTorch, no external downloads.
(require lispyn.macros [md])

(md "# Validation-guided training"
    "Define an early-stopping policy in Lisp. Checkpoints and real metrics stay available for inspection.")

(import torch)
(require lispyn.macros [with-seed])
(with-seed 42
  (setv model (torch.nn.Sequential
    (torch.nn.Linear 2 8) (torch.nn.ReLU) (torch.nn.Linear 8 2))))

(setv x (torch.tensor [[0.0 0.0] [0.2 0.1] [0.9 0.1] [0.8 0.2]
                        [0.1 0.9] [0.2 0.8] [1.0 1.0] [0.9 0.8]
                        [0.0 0.2] [0.9 0.0] [0.0 1.0] [1.0 0.9]]))
(setv y (torch.tensor [0 0 1 1 0 0 1 1 0 1 0 1]))
(setv train-data (torch.utils.data.TensorDataset
  (get x (slice 0 8)) (get y (slice 0 8))))
(setv validation (torch.utils.data.TensorDataset
  (get x (slice 8 None)) (get y (slice 8 None))))

(md "## Stop when validation stops improving"
    "Train for at most 40 epochs. After four epochs without a validation-loss improvement greater than 0.01, stop. The best and last checkpoints are saved separately.")

(require lispyn.macros [train-until])
(setv run
  (train-until model train-data 4
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 40
    :min-delta 0.01
    :metrics "accuracy"
    :batch-size 4
    :lr 0.02
    :device "cpu"
    :seed 42
    :experiment-dir "runs/early-stop"))
(print "Completed epochs:" (len (get run.history "train_loss")))
(print "Stopped early:" run.stopped-early)

(md "## Present the results"
    "Render the recorded training and validation losses in the shared Docusaurus-dark turquoise design.")

(require lispyn.macros [defcurve])
(defcurve training-chart :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :train-color "#40E0D0" :val-color "#8CF0EB")

(training-chart run :metric "loss"
  :title "Validation-guided training"
  :subtitle "Early stopping · Tiny offline classifier"
  :save "runs/early-stop/loss.png")
