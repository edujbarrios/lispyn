; Lispyn: evidence-backed, shareable experiments, written completely in Hy.
; These are illustrative scores, not metrics from a trained neural network.
(require lispyn.macros [md])
(md "# Reproducible evidence for an experiment"
    "**Source:** Hy/Lisp. **Output:** standard Jupyter, then portable Markdown."
    "**Data:** synthetic histories, not results from real model training."
    "Record hypotheses and checks explicitly so readers can verify each claim.")

(require lispyn.macros [experiment-params experiment-check checklist-report report-run])
(experiment-params
  (setv target-loss 0.75)
  (setv minimum-accuracy 0.8)
  (setv seed 42))

(setv history
  {"train_loss" [1.2 0.85 0.59]
   "val_loss" [1.3 0.86 0.68]
   "val_accuracy" [0.58 0.74 0.84]})
(setv final-loss (get (get history "val_loss") -1))
(setv final-accuracy (get (get history "val_accuracy") -1))

(setv checks
  [(experiment-check "Validation loss meets target"
     (<= final-loss target-loss)
     :observed final-loss :expected "at most target loss")
   (experiment-check "Validation accuracy meets target"
     (>= final-accuracy minimum-accuracy)
     :observed final-accuracy :expected "at least minimum accuracy")])
(checklist-report checks
  :title "Synthetic experiment evidence"
  :save "runs/synthetic_evidence.json")

(report-run history
  :title "Illustrative training summary"
  :notes "This example uses fixed synthetic numbers. Evaluate real models on separate held-out data before claiming performance.")

; The executed notebook can be published later, without rerunning training:
;   lispyn execute examples/evidence_story.hy
;   lispyn export examples/evidence_story.ipynb -o runs/evidence_story.md
