; Checks run before model training; direct Hy and notebooks use the same source.
(require lispyn.macros [md])
(md "# Check research inputs"
    "Catch unexpected shapes and non-finite features before training.")

(require lispyn.macros [assert-finite assert-shape])
(import numpy :as np)
(import pandas :as pd)

(setv features (np.array [[1.0 2.0] [3.0 4.0]]))
(assert-shape features [2 2] :name "features")
(assert-finite features :name "features")

(setv measurements (pd.DataFrame {"group" ["A" "B"] "score" [0.2 0.8]}))
(assert-shape measurements [2 None] :name "measurements")
(assert-finite (.to_numpy (get measurements "score")) :name "score")
(print "Research input checks passed.")
