; Lispyn 0.23.0a1 — leakage-aware time-series experiments in Hy.
; All samples are synthetic; no external data or GPU required.
(require lispyn.macros [md])

(md "# Chronological splits and walk-forward evaluation"
    "Never mix future observations into training windows."
    "Use a 2-sample embargo between training, validation and held-out test."
    "The test set stays untouched until the final evaluation.")

(require lispyn.macros
  [temporal-train-val-test temporal-folds cross-validate-temporal
   train-until save-cross-validation])
(import torch)

; A simple chronological regression dataset.
(setv x (torch.arange 45 :dtype torch.float32))
(setv y (+ (* x 0.25) 1.0))
(setv dataset (torch.utils.data.TensorDataset
  (.unsqueeze x 1) (.unsqueeze y 1)))

; Earlier time steps are training; the last are final testing samples.
(setv partitions (temporal-train-val-test dataset
  :validation-fraction 0.2 :test-fraction 0.15 :gap 2))
(setv history (get partitions 0))
(setv validation (get partitions 1))
(setv held-out-test (get partitions 2))
(print "History / validation / held-out test:"
  (len history) (len validation) (len held-out-test))

; Walk forward through history; never evaluate on future data.
(defn score-window [fold-train fold-valid fold repeat]
  ; Fresh parameters and native PyTorch optimizer each fold.
  (setv model (torch.nn.Linear 1 1))
  (setv result (train-until model fold-train 1
    :validation fold-valid
    :loss-fn (torch.nn.MSELoss)
    :epochs 2 :lr 0.001 :batch-size 8
    :seed (+ 42 fold)))
  (get (get result.history "val_loss") -1))

(setv report (cross-validate-temporal history score-window
  :n-splits 3 :test-size 4 :gap 1
  :min-train-size 8 :max-train-size 15 :metric "val_loss"))

(print "Mean walk-forward loss:" (get report "mean"))
(print "Embargo samples per fold:" (get report "gap"))
; Report only fold counts and summary metrics, not example data.
(save-cross-validation report "runs/temporal_cv_report.json")
