; Compare experimental evidence directly from Hy and Jupyter.
; This demonstration writes fixed SYNTHETIC histories; no model is trained.
(require lispyn.macros [md])
(md "# From experiments to a research leaderboard"
    "Compare recorded validation histories with no bespoke Python analysis."
    "**Important:** These values are synthetic and illustrative, not"
    "results from real training or evidence of model quality.")

(require lispyn.macros [experiment-leaderboard])
(import lispyn.reproducibility [write-json])

; Synthetic history.json files use the same scalar-only format as real runs.
; A real training run would instead specify :experiment-dir in train-until.
(write-json "runs/leaderboard-demo/baseline/history.json"
  {"train_loss" [1.2 0.9 0.75] "val_loss" [1.25 0.95 0.88]})
(write-json "runs/leaderboard-demo/tuned/history.json"
  {"train_loss" [1.1 0.8 0.56] "val_loss" [1.15 0.81 0.63]})
(write-json "runs/leaderboard-demo/compact/history.json"
  {"train_loss" [1.3 1.0 0.82] "val_loss" [1.38 1.08 0.92]})

(setv board
  (experiment-leaderboard "runs/leaderboard-demo"
    :metric "loss" :split "val" :rank-by "best"
    :save "runs/leaderboard-demo/leaderboard.md"))
(print "Ranked synthetic histories:" (get board "total"))

; CLI equivalent:
;   lispyn leaderboard examples/runs/leaderboard-demo --metric loss
;   lispyn leaderboard examples/runs/leaderboard-demo --output ranking.json
