; A tiny experiment. No downloaded dataset or GPU required.
; Lispyn expands the with-seed macro; all torch calls are native PyTorch.
(require lispyn.macros [with-seed])
(import torch)

(with-seed 42
  (setv x (torch.tensor [[1.0] [2.0] [3.0]]))
  (setv y (torch.tensor [[2.0] [4.0] [6.0]]))
  (setv model (torch.nn.Linear 1 1))
  (setv optimizer (torch.optim.SGD (.parameters model) :lr 0.05))
  (for [epoch (range 15)]
    (.zero_grad optimizer)
    (setv pred (model x))
    (setv loss (torch.nn.functional.mse_loss pred y))
    (.backward loss)
    (.step optimizer))
  (print "Final loss:" (.item loss)))
