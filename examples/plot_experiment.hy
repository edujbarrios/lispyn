; %% [markdown]
; ## Train once, visualize in Jupyter
; Toy classification experiment: no external datasets or pretrained weights.
; %% [code]
(import torch)
(import lispyn.workflow [train])

(torch.manual_seed 42)
(setv x (torch.tensor [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]]))
(setv y (torch.tensor [0 1 0 1]))
(setv data (torch.utils.data.TensorDataset x y))
(setv model (torch.nn.Linear 2 2))
(setv result (train model data
  :loss-fn (torch.nn.CrossEntropyLoss)
  :validation data
  :epochs 3
  :batch-size 2
  :device "cpu"
  :metrics "accuracy"
  :experiment-dir "runs/plot-demo"))
; %% [markdown]
; ## Training curves
; Figures become notebook outputs; files are also saved for sharing.
; %% [code]
(import lispyn.plots [plot-history])
(plot-history result :metric "loss" :save "runs/plot-demo/loss.png")
(plot-history result :metric "accuracy" :save "runs/plot-demo/accuracy.png")
(print "training plots saved")
