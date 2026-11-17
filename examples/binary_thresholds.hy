; %% [markdown]
; # Evaluate a binary AI classifier on validation data
; Scores are probabilities for the positive class (label 1). No model is trained.
(require lispyn.macros [threshold-report plot-threshold-metrics])
(setv true-labels [0 0 1 1 1 0 1 0])
(setv predicted-probabilities [0.1 0.35 0.4 0.8 0.7 0.9 0.55 0.2])
(setv evaluation
  (threshold-report true-labels predicted-probabilities
    :thresholds [0.3 0.5 0.75] :optimize "f1"
    :title "Synthetic validation threshold sweep"))
(assert (= (get evaluation "best_threshold") 0.3))
(assert (= (get evaluation "positives") 4))
(print "Threshold diagnostics passed!")
(plot-threshold-metrics true-labels predicted-probabilities
  :thresholds [0.3 0.5 0.75] :optimize "f1" :theme "paper")
