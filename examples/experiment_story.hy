; Lispyn: a readable, repeatable, shareable Hy experiment notebook.
; This example uses illustrative metric values; it does not train a model.
(require lispyn.macros [md])
(md "# A documented Lispyn experiment"
    "One Hy source gives you editable parameters, real Jupyter Markdown,"
    "and a shareable experiment report."
    "**Data:** illustrative metric histories only, not measured model results.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 4)
  (setv learning-rate 0.001))

(require lispyn.macros [report-run])
(setv history
  {"train_loss" [1.4 1.0 0.7 0.5]
   "val_loss" [1.5 1.1 0.75 0.8]
   "val_accuracy" [0.5 0.7 0.82 0.79]})

(report-run history
  :title "Illustrative training report"
  :notes "Synthetic histories, not a trained model. Compare training and validation before interpreting results."
  :save "runs/illustrative_experiment.md")
(print "Parameters:" seed epochs learning-rate)
