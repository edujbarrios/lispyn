; Lispyn 0.27: living experiment narratives authored in Hy/Lisp.
; This demonstration uses illustrative metrics rather than model results.
(require lispyn.macros [md])
(md "# A living Hy experiment narrative"
    "One Lisp file becomes a Jupyter notebook with measured claims."
    "**Demo values are synthetic and are not trained model results.**")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv target-accuracy 0.8)
  (setv seed 42))

; Each Jupyter code cell is independently compiled, so require again.
(require lispyn.macros [narrate metric-table experiment-check checklist-report])
(setv observed-accuracy 0.875)
(setv observed-loss 0.245)

(narrate
  "## Result\nWith seed {seed}, accuracy was **{accuracy:.1%}**, "
  :seed seed :accuracy observed-accuracy)

(narrate "The final loss was {loss:.3f}."
  :loss observed-loss :save "runs/narrative.md")

(metric-table
  {"val_accuracy" observed-accuracy
   "val_loss" observed-loss}
  :title "Illustrative validation metrics"
  :precision 4
  :save "runs/metric_table.md")

(setv check
  (experiment-check "Accuracy exceeds the target"
    (>= observed-accuracy target-accuracy)
    :observed observed-accuracy
    :expected target-accuracy))

(checklist-report [check]
  :title "Illustrative evidence"
  :save "runs/narrative_evidence.json")

; After executing:
; lispyn audit examples/living_narrative.ipynb --source examples/living_narrative.hy
; lispyn export examples/living_narrative.ipynb -o runs/living_narrative.md
