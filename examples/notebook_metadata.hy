; Notebook cell metadata demo: no downloads, pure Hy.
(require lispyn.macros [md raw-cell cell-tags])

(md "# Reproducible experiment parameters"
    "Use Lispyn to write parameterized notebook code with native Jupyter tags.")

(raw-cell "<!-- Raw cell: preserved for external renderers and exporters -->")

; Jupyter will recognize this as a parameter cell.
(require lispyn.macros [cell-tags])
(cell-tags "parameters")
(setv epochs 12)
(setv learning-rate 0.01)
(print "Configured epochs:" epochs "learning rate:" learning-rate)

(md "## Results"
    "Tag the next code cell as hidden input while keeping its real output.")

(require lispyn.macros [cell-tags])
(cell-tags "hide-input")
(print "Ready to run:" epochs "epochs")
