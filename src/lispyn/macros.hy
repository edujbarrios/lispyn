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
