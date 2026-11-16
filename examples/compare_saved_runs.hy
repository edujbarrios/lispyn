; A complete synthetic demonstration of reporting + plotting comparisons in Hy.
; The histories below are illustrative; no PyTorch model is trained here.
(require lispyn.macros [md])
(md "# Compare saved experiments in native Jupyter"
    "Use Lispyn macros to compare a baseline with a candidate run."
    "Synthetic metric histories demonstrate the workflow; no model was trained.")

(require lispyn.macros [compare-runs])
(import lispyn.reproducibility [write-json])
(write-json "runs/compare-demo/baseline/history.json"
  {"train_loss" [1.2 0.9 0.7] "val_loss" [1.25 0.95 0.85]})
(write-json "runs/compare-demo/candidate/history.json"
  {"train_loss" [1.1 0.75 0.5] "val_loss" [1.12 0.77 0.55]})
(setv paths ["runs/compare-demo/baseline" "runs/compare-demo/candidate"])
(setv comparison
  (compare-runs paths :metric "loss" :split "val" :baseline "baseline"
    :save "runs/compare-demo/comparison.md"))
(assert (= (get comparison "baseline") "baseline"))
(print "Compared saved histories:" (len (get comparison "runs")))

(require lispyn.macros [md])
(md "## Validation loss curves"
    "The figure comes directly from the two recorded histories.")

(require lispyn.macros [plot-runs])
(plot-runs paths :metric "loss" :split "val" :theme "paper"
  :title "Baseline vs candidate (synthetic)"
  :save "runs/compare-demo/comparison.png")
