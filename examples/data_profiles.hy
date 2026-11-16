; Data quality checks without printing individual observations.
(require lispyn.macros [md])
(md "# Data quality before training"
    "Aggregate NumPy and pandas checks without exporting dataset rows.")

(require lispyn.macros [array-profile table-profile])
(import numpy :as np)
(import pandas :as pd)

(setv features (np.array [[1.0 2.0] [3.0 (float "nan")] [5.0 (float "inf")]]))
(setv array-result (array-profile features :name "Sensor features"))
(assert (= (get array-result "nan") 1))
(assert (= (get array-result "infinite") 1))

(setv readings (pd.DataFrame {"subject" [1 2 2] "score" [0.5 None None]}))
(setv table-result (table-profile readings :name "Readings"))
(assert (= (get table-result "missing_cells") 2))
(assert (= (get table-result "duplicate_rows") 1))
(print "Data quality profiles complete.")
