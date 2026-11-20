; PyTorch diagnostics: compact, real CPU execution without downloads.
; %% [markdown]
; # Inspect, debug and evaluate a PyTorch model from Hy
; Build with lispyn build examples/torch_diagnostics.hy or run it directly.
; %% [code]
(require lispyn.macros [model-footprint])
(import torch)
(torch.manual_seed 7)
(setv model (torch.nn.Sequential (torch.nn.Linear 2 4)
                                (torch.nn.ReLU)
                                (torch.nn.Linear 4 2)))
(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv loader (torch.utils.data.DataLoader
               (torch.utils.data.TensorDataset x y) :batch_size 2))
(print "Nominal model footprint:" (model-footprint model))

; %% [markdown]
; ## Check backward gradients before updating parameters
; %% [code]
(require lispyn.macros [gradient-report assert-gradients with-anomaly-detection])
(setv optimizer (torch.optim.SGD (.parameters model) :lr 0.05))
(with-anomaly-detection
  (setv logits (model x))
  (setv loss ((torch.nn.CrossEntropyLoss) logits y))
  (.backward loss))
(setv gradients (gradient-report model))
(assert (get gradients "all_finite"))
(assert-gradients model)
(.step optimizer)
(.zero_grad optimizer)
(print "Gradient elements:" (get gradients "gradient_elements"))

; %% [markdown]
; ## Temporarily freeze parameters and restore the deterministic policy
; %% [code]
(require lispyn.macros [with-parameter-freeze with-determinism])
(setv first-parameter (next (iter (.parameters model))))
(assert first-parameter.requires_grad)
(with-parameter-freeze model
  (assert (not first-parameter.requires_grad)))
(assert first-parameter.requires_grad)
(with-determinism
  (assert (torch.are_deterministic_algorithms_enabled)))
(print "Trainable flags and policy restored")

; %% [markdown]
; ## Collect logits from a loader without training
; %% [code]
(require lispyn.macros [predict-batches])
(setv predictions (predict-batches model loader :device "cpu" :max-batches 2))
(assert (= (list predictions.shape) [4 2]))
(assert model.training)
(print "Batched inference:" (list predictions.shape))
