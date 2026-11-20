; Lispyn macro-first authoring: generated Python, not a hidden training engine.
; Compile: lispyn build examples/macro_first_training.hy
; Execute: lispyn execute examples/macro_first_training.hy
; Run: lispyn run examples/macro_first_training.hy
; %% [markdown]
; # Macros that generate reusable PyTorch code
; This example has no datasets to download and runs on a CPU.
; The training step and loop are created by Hy macros, not by a Python runner.
; %% [code]
(require lispyn.macros [deftrain-step collect-epochs optimizer-step])
(import torch)
(torch.manual_seed 13)
(setv model (torch.nn.Sequential (torch.nn.Linear 2 1)))
(setv optimizer (torch.optim.SGD (.parameters model) :lr 0.03))
(setv criterion (torch.nn.MSELoss))
(setv features (torch.tensor [[1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv labels (torch.tensor [[1.0] [0.0] [1.0]]))
(deftrain-step train-one model optimizer criterion)
(setv history
  (collect-epochs epoch 4
    (train-one features labels)))
(assert (= (len history) 4))
(setv final-loss
  (optimizer-step optimizer (criterion (model features) labels)))
(assert (not final-loss.requires_grad))
(print "Macro-generated training loss:" (list history))

; %% [markdown]
; ## Scoped modes are part of the generated program
; Both training and evaluation restore individual child-module states.
; %% [code]
(require lispyn.macros [with-training with-eval with-grad with-rng-fork
                       with-parameter-freeze with-determinism])
(.eval (get model 0))
(assert model.training)
(assert (not (get model 0).training))
(with-training model
  (assert model.training)
  (assert (get model 0).training)
  (assert (torch.is-grad-enabled)))
(assert model.training)
(assert (not (get model 0).training))
(with-eval model
  (assert (not model.training))
  (assert (not (torch.is-grad-enabled))))
(assert model.training)
(assert (not (get model 0).training))
(with-parameter-freeze model
  (assert (not (next (iter (.parameters model))).requires_grad)))
(assert (next (iter (.parameters model))).requires_grad)
(with [(torch.no_grad)]
  (assert (not (torch.is-grad-enabled)))
  (with-grad
    (assert (torch.is-grad-enabled))))
(setv rng-before (torch.get_rng_state))
(with-rng-fork
  (torch.manual_seed 999)
  (torch.randn [4]))
(assert (torch.equal rng-before (torch.get_rng_state)))
(with-determinism
  (assert (torch.are_deterministic_algorithms_enabled)))

; %% [markdown]
; ## Extend Lispyn with your own compile-time macros
; Define a custom macro and use it in the same notebook code cell.
; %% [code]
(defmacro assert-below [value limit]
  `(assert (< ~value ~limit)))
(assert-below (get history 0) 10.0)
(print "Custom Hy macro and generated PyTorch code passed")
