; Literal Markdown forms become real Markdown cells when built as a notebook.
; The same file also runs as an ordinary Hy program.
(require lispyn.macros [md markdown])

(md "# A notebook written in Lisp"
    "This section was written with the **md** macro, not a Python string printed in a cell.")

(setv values [2 4 6 8])
(print "Sample values:" values)

(markdown "## Reusable explanations"
          "The **Markdown** is preserved; the code remains executable Python after Hy compilation.")

(setv average (/ (sum values) (len values)))
(print "Average:" average)

(md "## Results"
    "You can use links, **bold** text, and code snippets like `torch.nn.Linear`.")
(print "Done.")
