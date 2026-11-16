; Real CPU PyTorch training on deliberately small SYNTHETIC data.
; Every notebook cell and the final figure are recorded, not fabricated.
(require lispyn.macros [md])
(md "# Quickstart · from Lisp to an executed research notebook"
    "A native PyTorch classifier on synthetic data. We reserve an unseen"
    "test partition for a final, illustrative evaluation.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 10))

(require lispyn.macros [md])
(md "## 1. Prepare data and isolate the test set"
    "Create two-dimensional toy samples; split them before training.")

(require lispyn.macros [train-val-test])
(import torch)
(torch.manual_seed seed)
(setv features
  (torch.tensor
    [[0.0 0.0] [0.0 1.0] [1.0 0.0] [1.0 1.0]
     [0.2 0.3] [0.3 0.9] [0.9 0.3] [0.8 0.9]
     [0.1 0.2] [0.1 0.8] [0.9 0.2] [0.9 0.8]
     [0.4 0.1] [0.4 0.9] [0.7 0.1] [0.7 0.9]]))
(setv labels (torch.tensor [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]))
(setv dataset (torch.utils.data.TensorDataset features labels))
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.25
    :test-fraction 0.25 :seed seed))
(print "Samples:" (len training) "train," (len validation)
  "validation," (len held-out) "held-out test")

(require lispyn.macros [md])
(md "## 2. Train a native PyTorch model"
    "Lispyn handles the training loop and early stopping; model, optimizer"
    "and tensors remain ordinary PyTorch objects.")

(require lispyn.macros [train-until])
(setv model (torch.nn.Linear 2 2))
(setv run
  (train-until model training 3
    :validation validation :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs epochs :batch-size 4 :lr 0.05 :seed seed :device "cpu"))
(print "Completed epochs:" (len (get run.history "train_loss")))

(require lispyn.macros [md])
(md "## 3. Report and evaluate"
    "Summarize recorded metrics, then evaluate the untouched test subset once."
    "This tiny synthetic dataset is a workflow demo, not research evidence.")

(require lispyn.macros [report-run narrate metric-table])
(import lispyn.training [evaluate-metrics])
(report-run run :title "PyTorch baseline · recorded metrics")
(setv held-out-loader (torch.utils.data.DataLoader held-out :batch_size 4))
(setv test-scores
  (evaluate-metrics run.model held-out-loader
    :loss-fn (torch.nn.CrossEntropyLoss) :metrics "accuracy" :device "cpu"))
(narrate "Held-out test accuracy: **{accuracy:.1%}**"
  :accuracy (get test-scores "accuracy"))
(metric-table
  {"best_val_loss" (min (get run.history "val_loss"))
   "test_loss" (get test-scores "loss")
   "test_accuracy" (get test-scores "accuracy")}
  :title "Recorded experiment scores" :precision 4)

(require lispyn.macros [md])
(md "## 4. Plot the real training history"
    "Generated from the run.history actually recorded above, never fake curves.")

(require lispyn.macros [curve])
(curve run :metric "loss" :theme "paper"
  :title "Quickstart · real PyTorch training"
  :subtitle "Train versus validation loss · synthetic dataset"
  :figsize [8 4.5]
  :save "../example_generated_notebooks/assets/quickstart_training_curves.png")
