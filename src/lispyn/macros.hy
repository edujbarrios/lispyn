; Lispyn macros. Copyright 2026 Eduardo J. Barrios.
; SPDX-License-Identifier: Apache-2.0

(defmacro with-seed [seed #* body]
  "Set Python and PyTorch random seeds, then evaluate BODY."
  (setv seed-symbol (hy.gensym "lispyn_seed"))
  `(do
    (import random)
    (import torch)
    (setv ~seed-symbol ~seed)
    (random.seed ~seed-symbol)
    (torch.manual_seed ~seed-symbol)
    ~@body))

(defmacro train-for [model loader optimizer criterion epochs]
  "Run a standard PyTorch fit loop with explicit components."
  `(do
     (import lispyn.training [fit])
     (fit ~model ~loader ~optimizer :loss-fn ~criterion :epochs ~epochs)))
