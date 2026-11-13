; Three-way dataset splitting and k-fold evaluation with synthetic samples.
; This is a small CPU-only demonstration; no real data is fetched.
(require lispyn.macros [md])
(md "# Train / validation / test and k-fold cross-validation"
    "Synthetic PyTorch dataset: every sample is used exactly once as validation"
    "across the 5 folds. Keep the test split separate until final evaluation.")

(require lispyn.macros [train-val-test cross-validation-folds])
(import torch)
(import lispyn.cross_validation [summarize-cv-scores])

(setv labels (+ (* [0] 15) (* [1] 15)))
(setv x (.unsqueeze (torch.arange 30) 1))
(setv data (torch.utils.data.TensorDataset x (torch.tensor labels)))
(setv partitions (train-val-test data :labels labels
                                 :validation-fraction 0.2
                                 :test-fraction 0.2 :seed 42))
(print "Train / val / test sizes:" (lfor part partitions (len part)))

; The independent held-out test set is not included in model selection.
(setv training (get partitions 0))
(setv fold-labels (lfor index (.indices training) (get labels index)))
(setv scores [])
(for [[fold-train fold-validation]
      (cross-validation-folds training :n-splits 5 :labels fold-labels :seed 42)]
  ; Replace this illustrative score with a fresh model trained on fold-train
  ; and evaluated only on fold-validation.
  (.append scores (/ (len fold-validation) (len training))))
(print (summarize-cv-scores scores :metric "validation_fraction"))
