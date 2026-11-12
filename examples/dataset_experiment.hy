; %% [markdown]
; ## A complete tiny PyTorch experiment
; This notebook is generated from Hy with Lispyn. No downloads or GPU needed.
; %% [code]
(import torch)
(import lispyn.data [read-dataset drop-missing train-validation-split
                     dataframe-to-dataset make-dataloader])
(setv frame (drop-missing (read-dataset "data/tiny_regression.csv")))
(setv partitions (train-validation-split frame :validation-fraction 0.25 :seed 42))
(setv training-frame (get partitions 0))
(setv validation-frame (get partitions 1))
(setv training-dataset (dataframe-to-dataset training-frame :features ["x"] :target "y"))
(setv validation-dataset (dataframe-to-dataset validation-frame :features ["x"] :target "y"))
(setv training-loader (make-dataloader training-dataset :batch-size 2 :shuffle True :seed 42))
(print "train rows:" (len training-frame) "validation rows:" (len validation-frame))
; %% [markdown]
; ## Training
; Use *native* PyTorch layers, loss functions and optimizers.
; %% [code]
(require lispyn.macros [with-seed])
(with-seed 42
  (setv model (torch.nn.Linear 1 1))
  (setv optimizer (torch.optim.SGD (.parameters model) :lr 0.01))
  (for [epoch (range 30)]
    (for [batch training-loader]
      (setv xs (get batch 0))
      (setv ys (get batch 1))
      (.zero_grad optimizer)
      (setv loss (torch.nn.functional.mse_loss (model xs) ys))
      (.backward loss)
      (.step optimizer))))
(print "training finished")
; %% [markdown]
; ## Evaluation
; Code cells share runtime variables after being compiled separately.
; %% [code]
(setv validation-loader (make-dataloader validation-dataset :batch-size 2))
(with [(torch.no_grad)]
  (for [batch validation-loader]
    (setv xs (get batch 0))
    (setv ys (get batch 1))
    (setv loss (torch.nn.functional.mse_loss (model xs) ys))
    (print "validation loss:" (.item loss))))
