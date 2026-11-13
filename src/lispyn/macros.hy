; Lispyn macros. Copyright 2026 Eduardo J. Barrios.
; SPDX-License-Identifier: Apache-2.0

(defmacro with-seed [seed #* body]
  "Seed Python, NumPy (when installed) and torch before evaluating BODY."
  `(do
    (import lispyn.reproducibility [seed-all])
    (seed-all ~seed)
    ~@body))

(defmacro train-for [model loader optimizer criterion epochs]
  "Run a standard PyTorch fit loop with explicit components."
  `(do
     (import lispyn.training [fit])
     (fit ~model ~loader ~optimizer :loss-fn ~criterion :epochs ~epochs)))

(defmacro with-eval [model #* body]
  "Temporarily enable evaluation/inference mode and restore training state."
  (setv model-symbol (hy.gensym "lispyn_model"))
  (setv previous-mode (hy.gensym "lispyn_training"))
  `(do
    (import torch)
    (setv ~model-symbol ~model)
    (setv ~previous-mode (getattr ~model-symbol "training"))
    (.eval ~model-symbol)
    (try
      (with [(torch.inference_mode)]
        ~@body)
      (finally
        (.train ~model-symbol ~previous-mode)))))

(defmacro curve [history #* options]
  "Generate a styled Matplotlib training curve from recorded metrics."
  `(do
    (import lispyn.plots [plot-history])
    (plot-history ~history ~@options)))

(defmacro defcurve [name #* options]
  "Define a reusable curve design; its arguments become default plot options."
  `(do
    (import lispyn.plots [make-plotter])
    (setv ~name (make-plotter ~@options))))

(defmacro md [#* content]
  "Document a notebook cell; Lispyn turns literal top-level calls into Markdown."
  None)

(defmacro markdown [#* content]
  "Long-form alias for md."
  None)

(defmacro head-tune [model data head #* options]
  "Train only named model head using the native PyTorch optimizer."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :head ~head ~@options)))

(defmacro full-tune [model data #* options]
  "Unfreeze all layers, then fine-tune with a fresh native optimizer."
  (setv model-symbol (hy.gensym "lispyn_model"))
  `(do
    (import lispyn.training [trainable])
    (import lispyn.workflow [train])
    (setv ~model-symbol ~model)
    (trainable ~model-symbol)
    (train ~model-symbol ~data ~@options)))
(defmacro ensure-package [requirement #* options]
  "Install missing named pip requirements when this cell is executed."
  `(do
    (import lispyn.dependencies)
    (lispyn.dependencies.ensure-package ~requirement ~@options)))

(defmacro use-package [requirement #* options]
  "Install a missing module if needed, then return it for ordinary Hy use."
  `(do
    (import lispyn.dependencies)
    (lispyn.dependencies.use-package ~requirement ~@options)))

(defmacro pip-install [#* requirements]
  "Generate an independent dependency cell; install only when executed."
  `(do
    (import lispyn.dependencies [ensure-packages])
    (ensure-packages ~@requirements)))

(defmacro compare-curves [runs #* options]
  "Compare matching metrics from multiple named training runs in one chart."
  `(do
    (import lispyn.plots [plot-comparison])
    (plot-comparison ~runs ~@options)))

(defmacro confusion-matrix [actual predicted #* options]
  "Plot classification labels as a styled confusion matrix."
  `(do
    (import lispyn.classification [plot-confusion-matrix])
    (plot-confusion-matrix ~actual ~predicted ~@options)))

(defmacro defconfusion [name #* options]
  "Create a reusable confusion-matrix design with default options."
  `(do
    (import lispyn.classification [make-confusion-plotter])
    (setv ~name (make-confusion-plotter ~@options))))

(defmacro train-until [model data patience #* options]
  "Train up to :epochs, stopping after PATIENCE epochs without improvement."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :patience ~patience ~@options)))

(defmacro raw-cell [#* content]
  "Emit a raw Jupyter cell from literal strings; no-op during Hy execution."
  None)

(defmacro cell-tags [#* tags]
  "Attach literal Jupyter metadata tags to the next code cell."
  None)

(defmacro classification-report [actual predicted #* options]
  "Compute precision, recall, F1 and support without scikit-learn."
  `(do
    (import lispyn.classification)
    (lispyn.classification.classification_report ~actual ~predicted ~@options)))

(defmacro train-accumulating [model data steps #* options]
  "Train with STEPS microbatches per optimizer update; forwards PyTorch options."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :accumulation-steps ~steps ~@options)))

(defmacro learning-rate-curve [history #* options]
  "Visualize a native PyTorch learning-rate scheduler as a Matplotlib figure."
  `(do
    (import lispyn.plots [plot-learning-rate])
    (plot-learning-rate ~history ~@options)))

(defmacro train-val-test [dataset #* options]
  "Create train, validation and test PyTorch Subsets with no data leakage."
  `(do
    (import lispyn.data [split-dataset-three])
    (split-dataset-three ~dataset ~@options)))

(defmacro cross-validation-folds [dataset #* options]
  "Yield seeded, optionally stratified or grouped PyTorch k-fold Subsets."
  `(do
    (import lispyn.data [k-fold-dataset])
    (k-fold-dataset ~dataset ~@options)))


; Cross-validation workflows are intentionally authored through Hy macros.

(defmacro tabular-train-val-test [frame #* options]
  "Split a pandas DataFrame into seeded train/validation/test partitions."
  `(do
    (import lispyn.data [train-validation-test-split])
    (train-validation-test-split ~frame ~@options)))

(defmacro tabular-cross-validation-folds [frame #* options]
  "Yield pandas (train, validation) folds with class/group isolation."
  `(do
    (import lispyn.data [k-fold-split])
    (k-fold-split ~frame ~@options)))

(defmacro repeated-cross-validation-folds [dataset #* options]
  "Yield (repeat, fold, train_subset, val_subset) for repeated k-fold."
  `(do
    (import lispyn.cv_workflows [repeated-k-fold-dataset])
    (repeated-k-fold-dataset ~dataset ~@options)))

(defmacro repeated-tabular-folds [frame #* options]
  "Yield (repeat, fold, train_frame, val_frame) for repeated k-fold."
  `(do
    (import lispyn.cv_workflows [repeated-k-fold-split])
    (repeated-k-fold-split ~frame ~@options)))

(defmacro cross-validate [dataset scorer #* options]
  "Run a Hy fold-scoring callback on fresh PyTorch CV partitions."
  `(do
    (import lispyn.cv_workflows [run-dataset-cross-validation])
    (run-dataset-cross-validation ~dataset ~scorer ~@options)))

(defmacro cross-validate-table [frame scorer #* options]
  "Run a Hy callback on fresh, optionally grouped pandas CV partitions."
  `(do
    (import lispyn.cv_workflows [run-frame-cross-validation])
    (run-frame-cross-validation ~frame ~scorer ~@options)))

(defmacro save-cross-validation [report path]
  "Save a privacy-minimal fold-score summary to a JSON file."
  `(do
    (import lispyn.cv_workflows [save-cv-report])
    (save-cv-report ~report ~path)))
