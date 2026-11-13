; %% [markdown]
; ## Reusable curve designs with Lisp macros
; This offline experiment demonstrates the same style across loss and accuracy.
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
  :experiment-dir "runs/curve-designs"))
; %% [markdown]
; ## Define once; reuse across metrics
; The defcurve and curve forms are real compile-time Hy macros.
; %% [code]
(require lispyn.macros [curve defcurve])

(defcurve report-curve
  :theme "paper"
  :train-color "#2563EB"
  :val-color "#D97706"
  :xlabel "Época"
  :train-label "Entrenamiento"
  :val-label "Validación")

(report-curve result :metric "loss"
  :title "Evolución de la pérdida"
  :ylabel "Entropía cruzada"
  :save "runs/curve-designs/loss.png")

(report-curve result :metric "accuracy"
  :title "Precisión del modelo"
  :ylabel "Exactitud"
  :save "runs/curve-designs/accuracy.svg")

(curve result
  :metric "loss"
  :theme "dark"
  :title "Vista oscura"
  :train-color "#22D3EE"
  :val-color "#FBBF24"
  :save "runs/curve-designs/dark.png")

(print "Saved three customizable curves")
