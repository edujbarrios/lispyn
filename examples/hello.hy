; Hello Lispyn: no torch, datasets or external files required.
(require lispyn.macros [md assert-close with-timing])

(md "# Hello Lispyn"
    "One tiny Hy program, three ways to run it: direct, build, execute.")

(print "Hello from Lispyn!")
(with-timing "Summing a few numbers"
  (setv numbers [2 3 5])
  (setv answer (sum numbers)))
(assert-close answer 10)
(print "2 + 3 + 5 =" answer)
(print "Lispyn checks passed!")
