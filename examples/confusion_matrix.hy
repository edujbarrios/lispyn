; Synthetic classification demo; no actual training or dataset required.
(require lispyn.macros [md pip-install])

(md "# Confusion matrix with Lispyn"
    "Create a reusable classification report using Lisp macros and the turquoise Docusaurus-dark theme. This example uses **synthetic labels**, not model predictions.")

(pip-install "matplotlib>=3.8")

(require lispyn.macros [defconfusion confusion-matrix])
(import matplotlib)
(matplotlib.use "Agg")

; 60 synthetic labeled examples, including predictable misclassifications.
(setv labels ["Cat" "Dog" "Bird" "Fox"])
(setv actual (lfor i (range 60) (get labels (% i 4))))
(setv predicted
  (lfor i (range 60)
    (get labels (if (= (% i 7) 0) (% (+ i 1) 4) (% i 4)))))

(defconfusion dark-classification
  :theme "docusaurus-dark"
  :palette "turquoise-spectrum"
  :figsize [8 6]
  :title-size 17
  :dpi 170)

(dark-classification actual predicted
  :classes labels
  :normalize "true"
  :title "Classification overview"
  :subtitle "Synthetic labels  ·  Docusaurus-dark  ·  Turquoise"
  :save "../docs/images/synthetic_confusion_matrix.png")

(print "Saved synthetic confusion-matrix figure.")

; Generate a precision / recall / F1 table from exactly these synthetic labels.
(require lispyn.macros [classification-report])
(setv scores
  (classification-report actual predicted
    :classes labels
    :save "../docs/reports/synthetic_classification_report.md"))
(print "Classification accuracy:" (get scores "accuracy"))
