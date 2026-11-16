# Lispyn

**Write machine-learning experiments in Hy, run native Python/PyTorch, and share executable Jupyter notebooks.**

Lispyn is an experimental tool for **Hy (Lisp for Python)**. Its macros simplify common research workflows without replacing PyTorch, pandas, Matplotlib or Jupyter.

[![PyPI](https://img.shields.io/pypi/v/lispyn)](https://pypi.org/project/lispyn/) [![Tests](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use Lispyn for research?

- **One experiment, multiple outputs.** Keep an experiment in one readable `.hy` source; run it in a terminal, compile a Jupyter notebook, or execute a notebook with saved outputs.
- **Less repetitive code.** Hy macros help with parameters, train/validation/test splits, cross-validation, PyTorch training, figures, and experiment comparisons.
- **Keep your existing tools.** Models, optimizers and data pipelines remain native Python/PyTorch, not a proprietary training engine.
- **Make work easier to review.** Generated notebooks preserve source provenance; reports, recorded metrics and optional integrity-checked experiment capsules help collaborators inspect and compare results.

Lispyn supports research workflows; it does **not** guarantee reproducibility or scientifically valid results. You still choose suitable datasets, splits, seeds and evaluation methods.

## Install and get the example

```bash
python -m pip install lispyn
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
```

The first command installs Lispyn, Hy, Jupyter execution support and Matplotlib. **PyTorch is installed separately** for training, with the right CPU/GPU build ([PyTorch installation guide](https://pytorch.org/get-started/locally/)). If working from a source checkout, `pip install -r requirements.txt` installs an optional research environment.

## Try it: one `hello.hy`, three ways to run it

The runnable file [`examples/hello.hy`](examples/hello.hy) needs **no PyTorch, dataset or GPU**:

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

Use the following commands **from the repository root**. Each runs or builds the **same source file**, but produces a different result.

**1. Run Hy directly — terminal output, no notebook**

```bash
lispyn run examples/hello.hy
```

Runs the Hy file immediately. Expect `Hello from Lispyn!`, `2 + 3 + 5 = 10`, `Lispyn checks passed!` and a timing line.

**2. Generate a Jupyter notebook — cells not executed**

```bash
lispyn build examples/hello.hy -o hello.ipynb
```

Creates `hello.ipynb` with Python code and a Markdown cell. **Nothing is executed**: there are no cell outputs.

**3. Generate and execute a Jupyter notebook — outputs included**

```bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

Creates `hello-executed.ipynb`, runs its code in a Python Jupyter kernel, and **saves the executed cells and printed results**. Open that file in your preferred Jupyter interface to inspect the output.

> **In short:** `run` = terminal; `build` = unexecuted notebook; `execute` = executed notebook. The `md` form becomes a notebook Markdown cell (and does nothing in direct Hy execution). Notebook code cells compile separately, so `require` a macro again in each cell that uses it. Only run or compile trusted Hy code: macros are not sandboxed.

## A real PyTorch experiment

The [full quickstart](examples/quickstart.hy) trains on tiny synthetic data, records metrics and makes a real loss plot. It is an illustration of the workflow, **not** a scientific benchmark.

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

Try the full example after installing a suitable PyTorch build:

```bash
lispyn execute examples/quickstart.hy
```

To generate the same notebook **without training**, use `lispyn build examples/quickstart.hy`.

## What else can you do?

Lispyn has **47 documented Hy macros** covering experiment parameters, dataset splits, cross-validation, training, comparisons, diagnostics, reports and plotting. See the [complete macro reference](supported_macros/README.md) and [examples](examples/).

For optional libraries, add an explicitly requested dependency cell:

```hy
(require lispyn.macros [pip-install])
(pip-install "scikit-learn>=1.5")
```

The package is checked or installed when the notebook **executes**, never during `build`.

## Share results with collaborators

For a **completed** experiment notebook, make an integrity-checked ZIP:

```bash
lispyn capsule examples/quickstart.hy example_generated_notebooks/quickstart_executed.ipynb -o study.zip
lispyn verify-capsule study.zip
```

Capsules include your Hy source and executed notebook, not raw datasets or model weights by default. **Review notebook outputs before sharing**; hashes check integrity, not authorship or research validity.

[Notebook gallery](example_generated_notebooks/) · [Macro reference](supported_macros/README.md) · [Citation](CITATION.cff)

*Created by Eduardo J. Barrios · [Apache-2.0](LICENSE) · Experimental alpha software.*
