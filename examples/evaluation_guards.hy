; Validate dataset isolation and model scores before evaluating metrics.
(require lispyn.macros [md])
(md "# Evaluation integrity"
    "Use stable sample identifiers and validate probability outputs.")

(require lispyn.macros [assert-disjoint assert-probabilities])
(import numpy :as np)

(setv train-ids ["train-1" "train-2" "train-3"])
(setv validation-ids ["valid-1" "valid-2"])
(setv test-ids ["test-1" "test-2"])
(assert-disjoint train-ids validation-ids)
(assert-disjoint train-ids test-ids :right-name "test")
(assert-disjoint validation-ids test-ids :left-name "validation" :right-name "test")

(setv binary-scores (np.array [0.1 0.6 0.9]))
(assert-probabilities binary-scores :name "positive-class scores")
(setv multiclass-scores (np.array [[0.8 0.2] [0.15 0.85]]))
(assert-probabilities multiclass-scores :name "class probabilities")
(print "Evaluation input guards passed.")
