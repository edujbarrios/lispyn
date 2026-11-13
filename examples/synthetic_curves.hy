; An attractive, reproducible Matplotlib demo; all values are SYNTHETIC.
(require lispyn.macros [md pip-install])

(md "# Training curves — Docusaurus dark"
    "A reusable **Lisp macro** creates a polished dark Matplotlib graphic using a turquoise palette. These are synthetic values, not the results of a trained model.")

(pip-install "matplotlib>=3.8")

(require lispyn.macros [defcurve compare-curves])
(import math)
(import matplotlib)
(matplotlib.use "Agg")

; No model, downloads, random seeds, or GPU required.
(setv epochs (list (range 1 49)))
(setv history
  {"train_loss"
     (lfor e epochs
       (+ 0.115 (* 1.61 (math.exp (- (/ e 13.0))))
          (* 0.020 (math.sin (* e 0.85)))))
   "val_loss"
     (lfor e epochs
       (+ 0.225 (* 1.68 (math.exp (- (/ e 14.9))))
          (* 0.040 (math.sin (/ e 2.6)))
          (* 0.012 (math.cos (* e 0.91)))))})

; Define this design once and reuse it for any recorded Lispyn run.
(defcurve turquoise-dark
  :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :figure-bg "#1B1B1D"
  :axes-bg "#23272F"
  :grid-color "#3A3F4B"
  :text-color "#E6EDF3"
  :spine-color "#4B5563"
  :legend-bg "#23272F"
  :legend-edge "#4B5563"
  :train-color "#40E0D0"
  :val-color "#8CF0EB"
  :train-label "Training · synthetic"
  :val-label "Validation · synthetic"
  :xlabel "Epoch"
  :ylabel "Cross-entropy loss"
  :title-size 18
  :label-size 12
  :tick-size 10
  :line-width 2.8
  :marker "."
  :alpha 0.96
  :fill-alpha 0.08
  :figsize [10 5.5]
  :dpi 180)

(turquoise-dark history
  :metric "loss"
  :title "Training dynamics"
  :subtitle "48 synthetic epochs  ·  Docusaurus dark  ·  Turquoise spectrum"
  :save "../docs/images/synthetic_training_curves.png")

(md "## Compare multiple experiments"
    "The **compare-curves** macro plots multiple named runs together using the same seven-color turquoise palette. These are also entirely synthetic.")

; Every notebook code cell is compiled separately, so require its macros again.
(require lispyn.macros [compare-curves])

(setv variants
  {"Baseline · synthetic"
    {"val_loss"
      (lfor e epochs
        (+ 0.36 (* 1.69 (math.exp (- (/ e 18.0))))
           (* 0.035 (math.cos (/ e 2.7)))))}
   "Tuned · synthetic"
    {"val_loss"
      (lfor e epochs
        (+ 0.18 (* 1.52 (math.exp (- (/ e 12.0))))
           (* 0.028 (math.sin (/ e 2.4)))))}
   "Augmented · synthetic"
    {"val_loss"
      (lfor e epochs
        (+ 0.25 (* 1.56 (math.exp (- (/ e 14.5))))
           (* 0.023 (math.cos (/ e 2.3)))))} })

(compare-curves variants
  :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :metric "loss"
  :split "val"
  :title "Comparing training configurations"
  :subtitle "Three simulated runs  ·  Shared design, automatic palette"
  :figsize [10 5.2]
  :linewidth 2.9
  :dpi 180
  :save "../docs/images/synthetic_comparison_curves.png")

(print "Created two turquoise-on-dark synthetic Matplotlib figures.")
