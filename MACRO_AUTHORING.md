# Write and compose macros with Lispyn

Lispyn is a **Hy macro library for AI experiment authoring**. Its product is
a vocabulary of compile-time forms, not an alternative Python tensor API or
a second training engine. The author controls native Python/PyTorch behavior;
macros transform repeated source patterns into Python that can be inspected
in a Jupyter notebook or with Hy's `hy2py`.

## What qualifies as a useful Lispyn macro?

Prefer macros that transform **structure**, not just rename a function:
- Bind runtime expressions exactly once, using `hy.gensym` for generated names.
- Introduce `try/finally` to preserve state even if user code raises.
- Synthesize ordinary Python loops, `def` functions or native context managers.
- Compose with other macros and **do not** hide the optimizer, model or data.
- Preserve a tiny runtime dependency surface. Native Python helpers remain
  appropriate for numerical algorithms and validation, not for simple syntax.

The v0.39.0a1 forms `deftrain-step`, `optimizer-step`,
`collect-epochs`, `with-training`, `with-grad` and
`with-rng-fork` expand directly to ordinary Python constructs. Three existing
context macros were also rewritten to expand their safety logic.

## Define your own macro

```hy
(defmacro assert-between [value lower upper]
  (setv v (hy.gensym "observed"))
  `(do
    (setv ~v ~value)
    (assert (<= ~lower ~v ~upper))))

(assert-between (+ 2 3) 0 10)
```

The macro is expanded during compilation; `hy.gensym` prevents its temporary
`observed` binding from colliding with a user's variables. Use `~` to
insert an expression and `~@` to splice a body of forms.

Unlike a Python function, a macro receives **code**, can introduce bindings
and control flow, and must be imported with `require`:

```hy
(require lispyn.macros [deftrain-step collect-epochs])
(deftrain-step train-one model optimizer criterion)
(setv history (collect-epochs epoch 4 (train-one inputs labels)))
```

`deftrain-step` generates a real Python function for tensor inputs and labels.
It calls the provided model, optimizer and criterion once per invocation;
it does not implement clipping, gradient accumulation, AMP, schedulers or
distributed training. For those behaviors, use explicit native PyTorch calls
and compose the matching dedicated macros.

## Compile, inspect, test

```bash
lispyn run examples/macro_first_training.hy
lispyn build examples/macro_first_training.hy -o macro-first.ipynb
lispyn execute examples/macro_first_training.hy -o macro-first-executed.ipynb
hy2py examples/macro_first_training.hy
pytest -q tests/test_macro_first.py
```

The notebook build contains compiled Python and Markdown cells. It does **not**
execute the model, but macros can execute code at **compile time**. Only build
trusted Hy sources. Never use untrusted Python callbacks or macro sources as a
security boundary.

Lispyn compiles notebook cells independently. Each cell using a Lispyn macro
must repeat its `require`. User-defined `defmacro` forms should be in the
**same code cell as their uses**; for reusable cross-cell macros, put them in an
importable `.hy` module and `require` them within each cell.

## Macro API versus Python backend

- **Macros:** source transformation, scoping and experiment composition.
- **Python helpers:** numerical validation, trained-model bookkeeping and
  complex data algorithms where plain Python is easier to audit.
- **Generated artifact:** plain Python/Jupyter, not an opaque Lispyn runtime.

The [complete macro catalog](supported_macros/README.md) documents every
supported macro. The [CPU-only example](examples/macro_first_training.hy)
demonstrates hygienic training-step generation, scope restoration and a
user-authored macro.
