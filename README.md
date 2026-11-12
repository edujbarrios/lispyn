# Lispyn

**Lisp macros for Python, data science, PyTorch and Jupyter.**

Lispyn is an early-stage, open-source experiment originally created by **Eduardo J. Barrios**. In an era when AI can generate new code quickly, Lispyn explores the complementary idea of *reusable abstractions*: define a pattern once as a Lisp macro, then use it across experiments.

Lispyn **extends [Hy](https://hylang.org/)** (a Lisp that runs on Python). It does not reinvent Python or restrict PyTorch. The first milestone focuses on converting trusted `.hy` programs to Python Jupyter notebooks and executing them optionally.

> Status: **v0.1.0a1 (experimental)**. No stability guarantees; not yet published on PyPI.

## Install from source

Requires Python 3.10+:

```bash
 git clone https://github.com/edujbarrios/lispyn.git
 cd lispyn
 python -m pip install -e '.[notebook,data]'
```

Install PyTorch separately using the [official selector](https://pytorch.org/get-started/locally/) for your CPU/GPU platform. The optional `data` extra installs pandas; for Parquet reading you may also need `pyarrow`.

## Quickstart

Open [`examples/first_experiment.hy`](examples/first_experiment.hy). It uses a tiny in-memory dataset and directly calls PyTorch. It also uses the Lispyn `with-seed` macro.

```bash
# Translate a trusted Hy program into a Python notebook, without executing it:
lispyn build examples/first_experiment.hy

# Translate, execute in Jupyter, and save outputs:
lispyn execute examples/first_experiment.hy

# Execute the Hy program directly (no notebook):
lispyn run examples/first_experiment.hy
```

Generated file: `examples/first_experiment.ipynb` (unless `-o` is specified). Generated notebooks start with an attribution cell naming the original author and project.

**Execution is explicit**: `build` never trains a model or executes Python cells. `execute` may take time and use a GPU; `run` executes the Hy source without a notebook. For `execute`, the notebook runs with its source file's directory as working directory, so relative dataset paths resolve predictably.

> **Security:** Hy macros can execute arbitrary code *during compilation*. `lispyn build` is **not** safe on untrusted `.hy` files. `execute` and `run` execute arbitrary Python/Hy code. Review experiment code and dependencies before use.

## Basic dataset loading

From a Hy file, call standard pandas, PyTorch, and other Python modules directly, or use Lispyn's small helpers:

```clojure
(import lispyn.data [read-dataset to-tensor])
(setv df (read-dataset "data/train.csv"))
(setv x (to-tensor (.to_numpy (.astype (get df ["feature-a" "feature-b"]) "float32"))))
```

`read-dataset` currently handles CSV, JSON and Parquet paths via pandas; `to-tensor` uses `torch.as_tensor`. Other data sources are available through their normal Python APIs.

## How this alpha version works

- **`lispyn build FILE.hy`**: calls Hy's `hy2py` compiler, then builds a valid Python 3 `.ipynb` with attribution and a **single code cell** containing the translated program. One code cell is deliberate: macros and definitions have one compilation context.
- **`lispyn execute FILE.hy`**: builds, runs the resulting notebook using `nbclient`, and saves results to `.ipynb`.
- **`lispyn run FILE.hy`**: delegates directly to the `hy` interpreter.
- **`with-seed`**: a starter macro that seeds Python's `random` and PyTorch, but **does not promise fully deterministic execution** across hardware or backends.

Lispyn doesn't yet generate multiple structured notebook sections, provide training DSLs or support arbitrary dataset schemas automatically. Those are candidates for future releases.

## Project roadmap

- [x] Package and command line
- [x] Hy-to-Jupyter notebook build (single Python cell)
- [x] Optional notebook execution
- [x] First macro and lightweight data helpers
- [ ] Notebook sections and markdown blocks
- [ ] Dataloader / training / evaluation macros
- [ ] Better diagnostics and reproducibility metadata

## Contributing

Development happens in public. Issues and pull requests are welcome. Start by keeping contributions small, tested, and documented. Install development dependencies with `python -m pip install -e '.[dev,data]'` and run `pytest`.

## License and attribution

Copyright © 2026 **Eduardo J. Barrios**. Licensed under the **Apache License 2.0**; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Redistributed copies and derivative works must preserve applicable copyright and attribution notices as described by the license. Apache 2.0 does **not** require credit for merely running Lispyn or using a generated notebook; generated notebooks include author attribution by default.