; Lispyn 0.22.0a1: Hy-first, repeated cross-validation with real training.
; Self-contained synthetic data, CPU-friendly, no downloads.
(require lispyn.macros [md])

(md "# Repeated, leakage-safe cross-validation with Lisp macros"
    "Train a fresh native PyTorch classifier for each validation fold."
    "Reserve the test set for *final* evaluation only, never for model selection.")

(require lispyn.macros [train-val-test cross-validate train-until save-cross-validation])
(import torch)

; A small binary classification dataset written entirely in Hy.
(setv features
  (torch.tensor (lfor i (range 40) [(/ i 40.0) 1.0]) :dtype torch.float32))
(setv labels (lfor i (range 40) (if (< i 20) 0 1)))
(setv dataset (torch.utils.data.TensorDataset
  features (torch.tensor labels :dtype torch.long)))

(setv partitions (train-val-test dataset
  :labels labels :validation-fraction 0.2 :test-fraction 0.1 :seed 42))
(setv training (get partitions 0))
(setv test-data (get partitions 2))
(setv train-labels
  (lfor index (getattr training "indices") (get labels index)))

; A *new* model and optimizer are constructed on every callback invocation.
(defn score-fold [fold-training fold-validation fold repeat]
  (setv model (torch.nn.Linear 2 2))
  (setv run (train-until model fold-training 2
    :validation fold-validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 4 :lr 0.05 :batch-size 8
    :seed (+ 42 (* repeat 100) fold)
    :metrics "accuracy"))
  (get (get run.history "val_accuracy") -1))

(setv report (cross-validate training score-fold
  :n-splits 4 :n-repeats 2 :labels train-labels
  :seed 42 :metric "val_accuracy"))

(print "Mean fold accuracy:" (get report "mean"))
(print "Validation-size weighted mean:" (get report "weighted_mean"))
(print "Test examples held out:" (len test-data))
; Save only fold scores, counts and aggregate statistics; no raw samples.
(save-cross-validation report "runs/repeated_cv_report.json")
