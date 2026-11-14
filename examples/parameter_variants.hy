; Reuse one readable Lispyn experiment in many configurations.
; This example uses synthetic histories only; no training or downloads.
(require lispyn.macros [md])
(md "# One Hy experiment, many reproducible notebooks"
    "Edit or override the parameters below without changing this source."
    "This example is intentionally synthetic: no model has been trained.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 5)
  (setv learning-rate 0.001)
  (setv hypothesis "a small learning rate yields stable loss"))

(require lispyn.macros [report-run])
(print "Run setup:" seed epochs learning-rate hypothesis)

; Only illustrative numbers, not actual model results.
(setv illustrative-loss (+ 0.5 learning-rate))
(setv history {"train_loss" [1.2 0.8 illustrative-loss]
               "val_loss" [1.3 0.9 (+ illustrative-loss 0.1)]})

(report-run history
  :title "Parameter variants (illustrative)"
  :notes hypothesis)

; To run a real experiment, replace illustrative histories with a
; native PyTorch train-until result and report-run its real history.
