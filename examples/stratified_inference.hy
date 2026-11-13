; %% [markdown]
; ## Balanced classification with a real Lisp macro
; An offline, tiny example: no images or external model downloads.
; %% [code]
(import torch)
(import lispyn.data [split-dataset])
(import lispyn.workflow [train])
(require lispyn.macros [with-eval])

(torch.manual_seed 42)
(setv xs (torch.randn 20 2))
(setv labels (torch.tensor [0 0 0 0 0 0 0 0 0 0
                            1 1 1 1 1 1 1 1 1 1]))
(setv data (torch.utils.data.TensorDataset xs labels))
(setv partitions (split-dataset data :labels labels
                                :validation-fraction 0.2 :seed 42))
(setv training (get partitions 0))
(setv validation (get partitions 1))
(setv model (torch.nn.Linear 2 2))
(setv result
  (train model training
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 2
    :batch-size 4
    :metrics "accuracy"
    :device "cpu"))
(print "validation accuracy:" (get result.history "val_accuracy"))
; %% [markdown]
; ## Inference: no gradients and no manual state restoration
; %% [code]
(.train result.model)
(with-eval result.model
  (assert (not result.model.training))
  (assert (not (torch.is-grad-enabled)))
  (setv prediction (result.model (torch.randn 1 2)))
  (print "inference prediction shape:" prediction.shape))
(assert result.model.training)
(print "original train mode restored")
