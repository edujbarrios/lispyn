; Native PyTorch gradient accumulation and learning-rate scheduling.
(require lispyn.macros [md])

(md "# Advanced training with Lispyn"
    "Use gradient accumulation, global gradient clipping, and a cosine learning-rate scheduler in a real CPU PyTorch experiment. This tiny classification dataset is synthetic; training results are real and deterministic when run.")

(import torch)
(torch.manual_seed 21)
(setv model (torch.nn.Sequential
  (torch.nn.Linear 2 12)
  (torch.nn.ReLU)
  (torch.nn.Linear 12 2)))

; Small offline dataset: 12 labeled points.
(setv x (torch.tensor [[0.0 0.0] [0.2 0.1] [0.9 0.1] [0.8 0.2]
                       [0.1 0.9] [0.2 0.8] [1.0 1.0] [0.9 0.8]
                       [0.0 0.2] [0.9 0.0] [0.0 1.0] [1.0 0.9]]))
(setv y (torch.tensor [0 0 1 1 0 0 1 1 0 1 0 1]))
(setv training (torch.utils.data.TensorDataset
  (get x (slice 0 8)) (get y (slice 0 8))))
(setv validation (torch.utils.data.TensorDataset
  (get x (slice 8 None)) (get y (slice 8 None))))

(md "## Accumulate gradients and schedule the learning rate"
    "The train-accumulating Hy macro keeps native PyTorch optimizers. Three microbatches contribute to each optimizer update. Gradient norms are clipped at 1.0, and the cosine scheduler advances once per epoch.")

(require lispyn.macros [train-accumulating])
(setv run
  (train-accumulating model training 3
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :metrics "accuracy"
    :epochs 12
    :batch-size 1
    :lr 0.025
    :gradient-clip 1.0
    :scheduler "cosine"
    :scheduler-kwargs {"eta_min" 0.001}
    :device "cpu"
    :seed 21
    :experiment-dir "runs/advanced-training"))

(print "Epochs:" (len (get run.history "train_loss")))
(print "Recorded learning rates:" (get run.history "learning_rate"))

(md "## Plot training and learning-rate curves"
    "The plots share a Docusaurus-inspired dark background and a reusable turquoise palette. Points come from an actual offline training run, not fabricated training results.")

(require lispyn.macros [defcurve learning-rate-curve])
(defcurve dark-curves
  :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :train-color "#40E0D0"
  :val-color "#8CF0EB")

(dark-curves run
  :metric "loss"
  :title "Accumulated PyTorch training"
  :subtitle "12 epochs · CPU · 3 microbatches/update"
  :save "runs/advanced-training/loss.png")

(learning-rate-curve run
  :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :title "Cosine learning-rate schedule"
  :subtitle "Recorded optimizer LR at each epoch"
  :save "runs/advanced-training/learning_rate.png")
