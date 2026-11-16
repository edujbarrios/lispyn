; A small classification dataset, ready to turn into an AI notebook.
(require lispyn.macros [md])
(md "# Inspect the classes before training"
    "A label-frequency report and figure from the same input labels.")

(require lispyn.macros [label-balance plot-label-balance])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv report (label-balance labels :title "Training label distribution"))
(assert (= (get report "total") 6))
(assert (= (get report "num_classes") 3))
(print "Class balance checks passed!")
(plot-label-balance labels :title "Training label distribution" :theme "paper")
