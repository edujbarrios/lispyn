# Lispyn

**Write experiments in Hy (Lisp for Python). Run them directly or turn them into Jupyter notebooks—with or without executed outputs.**

Lispyn adds concise research macros to **native Python, PyTorch and Jupyter**; it does not replace those tools.

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why Lispyn for research?

- **Write once, choose how to run:** terminal script, unexecuted notebook or notebook with saved outputs.
- **Reduce boilerplate:** macros for experiment parameters, PyTorch training, dataset splits, cross-validation, plots and comparisons.
- **Share inspectable results:** standard Python notebook cells, source provenance, metric reports and optional integrity-checked experiment archives.

You still control your models, datasets and evaluation methods. Lispyn cannot guarantee reproducibility or scientific validity.

## Install

```bash
pip install lispyn
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
```

Hy, Jupyter execution and Matplotlib are included. **PyTorch is installed separately** for your CPU/GPU setup ([official instructions](https://pytorch.org/get-started/locally/)). From a clone, `pip install -r requirements.txt` provides an optional research environment.

## Quick start: the same `hello.hy` in three modes

[`examples/hello.hy`](examples/hello.hy) runs without PyTorch or a dataset:

```hy
(require lispyn.macros [md])
(md "# Hello Lispyn"
    "A tiny Hy program you can run directly or as a Jupyter notebook.")

(require lispyn.macros [assert-close with-timing])
(print "Hello from Lispyn!")
(with-timing "Summing a few numbers"
  (setv numbers [2 3 5])
  (setv answer (sum numbers)))
(assert-close answer 10)
(print "2 + 3 + 5 =" answer)
(print "Lispyn checks passed!")
```

Run these commands **from the repository root**:

**1. Run Hy directly** — prints to the terminal; creates no notebook.

```bash
lispyn run examples/hello.hy
```

**2. Generate a Jupyter notebook (not executed)** — writes `hello.ipynb` with Markdown and Python code cells. **Nothing is executed**; no outputs are saved.

```bash
lispyn build examples/hello.hy -o hello.ipynb
```

**3. Generate and execute a Jupyter notebook** — writes `hello-executed.ipynb` and **saves the executed cells** with their outputs.

```bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

Check for `2 + 3 + 5 = 10` and `Lispyn checks passed!` in terminal output (mode 1) or notebook outputs (mode 3). The timing value varies. The `md` macro produces a real Markdown cell in notebooks. Macros are not sandboxed: **use trusted Hy sources**.

## Real research workflow: native PyTorch

The [runnable quickstart](examples/quickstart.hy) trains on a small synthetic dataset, records metrics and generates an actual training plot. It is a workflow example, not a benchmark.

<!-- QUICKSTART_EXECUTED:START -->
**Hy syntax** — an excerpt from the [full runnable experiment](examples/quickstart.hy) (native PyTorch; tiny synthetic dataset):

```hy
(require lispyn.macros [train-until])
(setv model (torch.nn.Linear 2 2))
(setv run
  (train-until model training 3
    :validation validation :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs epochs :batch-size 4 :lr 0.05 :seed seed :device "cpu"))
(print "Completed epochs:" (len (get run.history "train_loss")))

(require lispyn.macros [curve])
(curve run :metric "loss" :theme "paper"
  :title "Quickstart · real PyTorch training"
  :subtitle "Train versus validation loss · synthetic dataset"
  :figsize [8 4.5]
  :save "../example_generated_notebooks/assets/quickstart_training_curves.png")
```

**Real recorded output:**

`Samples: 8 train, 4 validation, 4 held-out test` · `Completed epochs: 10`

Held-out test accuracy: **50.0%**

![Actual training and validation loss from the Hy quickstart](example_generated_notebooks/assets/quickstart_training_curves.png)

[View every executed Jupyter cell](example_generated_notebooks/quickstart_executed.ipynb) · [Full Hy source](examples/quickstart.hy)

_Toy data for workflow demonstration only; not a scientific benchmark._
<!-- QUICKSTART_EXECUTED:END -->

To run the example yourself (after installing PyTorch):

```bash
lispyn execute examples/quickstart.hy
```

Use `lispyn build examples/quickstart.hy` to generate the notebook without training.

## More tools and sharing

Lispyn includes **47 documented Hy macros**. Browse the [macro reference](supported_macros/README.md), [Hy examples](examples/) and [notebook gallery](example_generated_notebooks/).

Optional dependencies can be declared as an execution-time notebook cell, never installed during `build`:

```hy
(require lispyn.macros [pip-install])
(pip-install "scikit-learn>=1.5")
```

Package and verify a **completed** experiment without rerunning it:

```bash
lispyn capsule examples/quickstart.hy example_generated_notebooks/quickstart_executed.ipynb -o study.zip
lispyn verify-capsule study.zip
```

Capsules omit raw datasets and weights by default; review saved notebook outputs before sharing. Checksums verify integrity, not authorship.

[How to cite Lispyn](CITATION.cff) · [Apache-2.0](LICENSE) · *Created by Eduardo J. Barrios; experimental alpha software.*
