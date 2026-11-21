; The front-page showcase: optional pip cell, native CPU PyTorch, and custom plots.
; Run: lispyn run examples/showcase.hy
; Build: lispyn build examples/showcase.hy -o showcase.ipynb
; Execute: lispyn execute examples/showcase.hy -o showcase-executed.ipynb

; %% [markdown]
; # From Hy macros to real Python models and plots
; All data is synthetic, all training is CPU-only, and nothing is downloaded.

; %% [code]
(require lispyn.macros [pip-install])
(pip-install "matplotlib>=3.8")

; %% [markdown]
; ## Configure a fresh PyTorch model, optimizer and training step
; PyTorch must be installed separately using your platform's supported build.

; %% [code]
(require lispyn.macros [defmodel defoptimizer deftrain-step collect-epochs])
(import torch)
(torch.manual_seed 42)

(defmodel make-mlp [features hidden]
  (torch.nn.Linear features hidden)
  (torch.nn.ReLU)
  (torch.nn.Linear hidden 1))
(setv model (make-mlp 2 8))
(defoptimizer make-adam [model rate] torch.optim.Adam :weight-decay 0.01)
(setv optimizer (make-adam model 0.05))
(setv criterion (torch.nn.MSELoss))
(setv samples (torch.tensor [[0.0 0.0] [0.0 1.0] [1.0 0.0] [1.0 1.0]]))
(setv targets (torch.tensor [[0.0] [1.0] [1.0] [2.0]]))
(deftrain-step train-one model optimizer criterion)
(setv losses (collect-epochs epoch 4
  (train-one samples targets)))
(assert (= (len losses) 4))
(print "Training losses:" losses)

; %% [markdown]
; ## Generate a reusable plot function from Hy
; Native Matplotlib remains fully accessible.

; %% [code]
(require lispyn.macros [defplot with-figure])
(import matplotlib.pyplot :as plt)

(defplot loss-chart [losses] [fig ax]
  (.plot ax (range (len losses)) losses :label "Training loss" :marker "o")
  (.set_title ax "Training progress")
  (.set_xlabel ax "Epoch")
  (.set_ylabel ax "MSE")
  (.legend ax))
(setv figure (loss-chart losses))
(.savefig figure "showcase-loss.png")
(plt.close figure)

(setv comparison (with-figure [fig ax]
  (.bar ax ["First" "Last"] [(get losses 0) (get losses -1)])
  (.set_title ax "Training loss at a glance")))
(.savefig comparison "showcase-comparison.png")
(plt.close comparison)
(print "Showcase generated two PNG charts")
