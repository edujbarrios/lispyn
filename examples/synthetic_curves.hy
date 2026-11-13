; A tiny reproducible visualization: synthetic values, NOT model results.
(require lispyn.macros [md pip-install])

(md "# Plot training curves with Lispyn"
    "An example using **synthetic data** to demonstrate Matplotlib styling. No model training or datasets are required.")

(pip-install "matplotlib>=3.8")

(require lispyn.macros [defcurve])
(import matplotlib)
(matplotlib.use "Agg")

(setv history {"train_loss" [1.62 1.40 1.24 1.11 1.02 0.91
                             0.82 0.74 0.67 0.61 0.55 0.50
                             0.46 0.42 0.38 0.35 0.32 0.29]
               "val_loss" [1.66 1.46 1.29 1.18 1.10 1.00
                            0.95 0.88 0.84 0.79 0.77 0.72
                            0.73 0.70 0.72 0.69 0.70 0.68]})

(defcurve my-curves
  :theme "paper"
  :train-color "#2563EB"
  :val-color "#D97706"
  :xlabel "Epoch"
  :ylabel "Loss"
  :figsize [9 4.5]
  :dpi 160)

(setv figure
  (my-curves history
    :title "Training and validation loss · synthetic example"
    :save "../docs/images/synthetic_training_curves.png"))
(print "Saved docs/images/synthetic_training_curves.png")
