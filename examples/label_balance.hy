; A small classification dataset, ready to turn into an AI notebook.
(require lispyn.macros [md])
(md "# Inspect the classes before training"
    "A label-frequency report and figure from the same input labels.")

(require lispyn.macros [label-balance plot-label-balance class-weights])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv report (label-balance labels :title "Training label distribution"))
(assert (= (get report "total") 6))
(assert (= (get report "num_classes") 3))
(setv balanced (class-weights labels :class-order ["bird" "cat" "dog"]))
(assert (= (get balanced "class_order") ["bird" "cat" "dog"]))
(assert (= (get balanced "counts") [1 3 2]))
(assert (= (get balanced "weights") [2.0 (/ 2 3) 1.0]))
(print "Class balance checks passed!")
(plot-label-balance labels :title "Training label distribution" :theme "paper")
