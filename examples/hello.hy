; Hello Lispyn: runs without PyTorch or datasets.
(require lispyn.macros [md])
(md "# Hello Lispyn"
    "A tiny Hy program you can run directly or as a Jupyter notebook.")

; Lispyn compiles notebook cells independently: require macros in each code cell.
(require lispyn.macros [assert-close with-timing])
(print "Hello from Lispyn!")
(with-timing "Summing a few numbers"
  (setv numbers [2 3 5])
  (setv answer (sum numbers)))
(assert-close answer 10)
(print "2 + 3 + 5 =" answer)
(print "Lispyn checks passed!")
