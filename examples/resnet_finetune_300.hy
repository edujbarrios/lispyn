; ResNet18 experiment: all results are Python/Jupyter notebook cells.
; Required actual images: examples/data/resnet_images/class_name/*.jpg or *.png
; This source is NOT executed in CI (300 epochs need a real dataset/GPU).
(require lispyn.macros [md])

(md "# ResNet18 — transfer learning + full fine-tuning"
    "An end-to-end **PyTorch** experiment written in Lispyn. It loads labeled images, splits by class, runs head-only tuning, then unfreezes the backbone for **300 epochs**, and produces publication-ready curves."
    "**Author:** Eduardo J. Barrios · **Source:** Lispyn.")

(import torch)
(import torchvision)
(import lispyn.media [image-folder])
(import lispyn.data [split-dataset])
(import lispyn.reproducibility [seed-all])

(seed-all 42)
(setv weights torchvision.models.ResNet18_Weights.DEFAULT)
(setv dataset (image-folder "data/resnet_images" :transform (weights.transforms)))
(setv groups (split-dataset dataset :stratify True :validation-fraction 0.2 :seed 42))
(setv training (get groups 0))
(setv validation (get groups 1))
(setv model (torchvision.models.resnet18 :weights weights))
(setv model.fc (torch.nn.Linear model.fc.in_features (len dataset.classes)))
(print "Classes:" dataset.classes)
(print "Train/validation:" (len training) (len validation))

(md "## Phase 1 — classifier fine-tuning"
    "Freeze the pretrained ResNet18 backbone and optimize only `model.fc` for **10 epochs**. The class split is reproducible and separate from the training set."
    "**Note:** This example needs at least two labeled images per class. For correlated images from the same subject, use a *group-aware split* instead.")

(require lispyn.macros [head-tune])
(setv head-run
  (head-tune model training "fc"
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 10
    :batch-size 16
    :lr 0.001
    :seed 42
    :metrics "accuracy"
    :experiment-dir "runs/resnet/head"))
(print "Head validation accuracy:" (get head-run.history "val_accuracy"))

(md "## Phase 2 — deep/full fine-tuning: 300 epochs"
    "Unfreeze **every ResNet18 layer**, create a new optimizer with a smaller learning rate, and train for **300 epochs**. This is full fine-tuning of pretrained weights, **not** training from scratch."
    "A real 300-epoch run may take hours and require a GPU. Run it only with your own dataset and appropriate resources.")

(require lispyn.macros [full-tune])
(setv deep-run
  (full-tune head-run.model training
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 300
    :batch-size 16
    :lr 0.00001
    :seed 42
    :metrics "accuracy"
    :experiment-dir "runs/resnet/deep"))
(print "Full tuning best accuracy:" (max (get deep-run.history "val_accuracy")))

(md "## Curves and export"
    "Use one Lisp macro to define a consistent Matplotlib style for **both** training curves. The best checkpoints and JSON histories are saved under `runs/resnet/`.")

(require lispyn.macros [defcurve])
(defcurve resnet-curve
  :theme "paper"
  :train-color "#2563EB"
  :val-color "#D97706"
  :xlabel "Epoch"
  :train-label "Train"
  :val-label "Validation"
  :figsize [9 4])

(resnet-curve deep-run :metric "loss"
  :title "ResNet18 — 300-epoch loss"
  :ylabel "Cross-entropy loss"
  :save "runs/resnet/deep/loss.png")

(resnet-curve deep-run :metric "accuracy"
  :title "ResNet18 — 300-epoch accuracy"
  :ylabel "Top-1 accuracy"
  :save "runs/resnet/deep/accuracy.png")
(print "Saved ResNet18 plots, histories, and best checkpoints.")
