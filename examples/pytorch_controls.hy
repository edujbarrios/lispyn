; Native PyTorch controls with Hy macros. CPU-only, no downloads.
; %% [markdown]
; # PyTorch controls: one Hy source, a Python notebook or direct execution
; Build: lispyn build examples/pytorch_controls.hy
; Run:   lispyn run examples/pytorch_controls.hy
; %% [code]
(require lispyn.macros [with-seed with-no-grad with-autocast
  clip-gradients freeze-module unfreeze-module parameter-count
  move-to-device save-model load-model])
(import torch)
(with-seed 11
  (setv model (torch.nn.Linear 2 1)))
(setv sample {"x" (torch.tensor [[1.0 2.0] [3.0 4.0]])
             "target" (torch.tensor [[0.0] [1.0]])})
(setv batch (move-to-device sample "cpu"))
(setv optimizer (torch.optim.SGD (.parameters model) :lr 0.01))
(print "Parameter counts:" (parameter-count model))

; %% [markdown]
; ## Forward, backward, clip, update
; Autocast wraps the forward pass only. Backward and clipping remain explicit.
; %% [code]
(setv prediction (with-autocast "cpu" torch.bfloat16
                   (model (get batch "x"))))
(setv loss ((torch.nn.MSELoss) prediction (get batch "target")))
(.backward loss)
(setv original-norm (clip-gradients model 1.0))
(.step optimizer)
(.zero_grad optimizer)
(print "Gradient norm before clipping:" original-norm)

; %% [markdown]
; ## Freeze and evaluate, then restore for training
; Freezing controls parameter gradients, not batch-normalization/dropout modes.
; %% [code]
(freeze-module model)
(assert (= (get (parameter-count model) "trainable") 0))
(setv output (with-no-grad (model (get batch "x"))))
(assert (not output.requires_grad))
(unfreeze-module model)
(assert (> (get (parameter-count model) "trainable") 0))

; %% [markdown]
; ## Save and restore native state dictionaries
; Load only checkpoint files you trust, even with weights-only loading.
; %% [code]
(save-model model "artifacts/controls.pt" :optimizer optimizer
  :history {"train_loss" [(float (.item loss))]})
(setv history (load-model model "artifacts/controls.pt" :optimizer optimizer))
(assert (= (len (get history "train_loss")) 1))
(print "PyTorch controls example passed")
