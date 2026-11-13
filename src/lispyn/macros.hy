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
