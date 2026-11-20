# Lispyn — Lisp-powered AI notebooks

**Write machine-learning experiments in Hy (Lisp for Python), then run them as scripts or turn them into Jupyter notebooks with ordinary Python code and real execution outputs.**

Lispyn is a small **experiment authoring and notebook workflow layer**, not a replacement for PyTorch, NumPy, pandas, or Matplotlib. You keep the same models, data, optimizers and Python ecosystem; Hy macros make repeated experiment steps easier to express, review and reuse.

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use Lispyn for AI notebooks?

**One source, three useful artifacts.** Develop a `.hy` experiment once. Execute it as a script (`run`), compile a reviewable but unexecuted notebook (`build`), or produce a notebook with saved outputs (`execute`). Unlike a notebook-only workflow, you can version-control readable source while sharing standard `.ipynb` files.

**Less repeated experiment wiring, without a new tensor API.** The [real PyTorch quickstart](examples/quickstart.hy) splits a dataset into train, validation and held-out test subsets:

~~~hy
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.25
    :test-fraction 0.25 :seed seed))
~~~

Lispyn's training macros also support early stopping, metrics, gradient accumulation, checkpoints and learning-rate schedules; their results are ordinary PyTorch-backed objects.

**Explicit diagnostics before and after training.** The [input validation example](examples/validate_inputs.hy) rejects unexpected shapes and nonfinite values:

~~~hy
(require lispyn.macros [assert-finite assert-shape])
(import numpy :as np)
(import pandas :as pd)

(setv features (np.array [[1.0 2.0] [3.0 4.0]]))
(assert-shape features [2 2] :name "features")
(assert-finite features :name "features")
~~~

Check class imbalance using the [runnable example](examples/label_balance.hy):

~~~hy
(require lispyn.macros [label-balance plot-label-balance])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv report (label-balance labels :title "Training label distribution"))
~~~

For classification, `class-weights` derives inverse-frequency loss weights, and `threshold-report` compares binary probability cutoffs **on validation data**, not the held-out test set.

**Readable, evidence-linked notebooks.** Generate metric tables, plots, comparisons and Markdown from recorded runs rather than manually pasting numbers. The [quickstart](examples/quickstart.hy) reports PyTorch results directly:

~~~hy
(require lispyn.macros [report-run narrate metric-table])
(import lispyn.training [evaluate-metrics])
(report-run run :title "PyTorch baseline · recorded metrics")
~~~

See its [executed notebook](example_generated_notebooks/quickstart_executed.ipynb), or use parameterized experiments, sweeps, notebook auditing and export to Markdown.

**Native PyTorch controls when you write your own loop.** The [CPU-only example](examples/pytorch_controls.hy) demonstrates `with-no-grad`, `with-autocast`, `clip-gradients`, `freeze-module`, `unfreeze-module`, `parameter-count`, `move-to-device`, `save-model` and `load-model`. These are focused helpers, not a hidden training engine.

**Model diagnostics and batched inference.** The [diagnostics example](examples/torch_diagnostics.hy) checks for missing, NaN or infinite gradients before optimizer updates; estimates parameter and buffer storage; temporarily freezes parameters and enables deterministic algorithms; and performs loader-based inference while restoring the model's train/eval state.

~~~hy
(require lispyn.macros [gradient-report assert-gradients])
(setv gradients (gradient-report model))
(assert (get gradients "all_finite"))
(assert-gradients model)
~~~

Use `model-footprint` for an **estimate of parameter and buffer storage**, not GPU peak memory. Use `predict-batches` to collect predictions from a bounded number of batches; it keeps predictions in memory, so avoid unbounded runs on large datasets. See [all seven new diagnostics macros](supported_macros/README.md#pytorch-diagnostics--inference).

## Install

~~~bash
pip install lispyn
~~~

The base package includes Hy, notebook compilation/execution and Matplotlib. NumPy is available through Matplotlib; add pandas with `pip install "lispyn[data]"`. **PyTorch is installed separately**: select the appropriate CPU/GPU build from [PyTorch's installer](https://pytorch.org/get-started/locally/). A cloned repository provides a broader optional development setup:

~~~bash
pip install -r requirements.txt
~~~

Python 3.10+ is required. Lispyn is experimental alpha software; macros and APIs may change.

## Quickstart: run, build, execute

Clone the repo for the sample (no dataset or PyTorch required):

~~~bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
~~~

**1. Run directly in a terminal.** No notebook is written.

~~~bash
lispyn run examples/hello.hy
~~~

**2. Build an ordinary Jupyter notebook.** Nothing is executed; `hello.ipynb` contains compiled Python and Markdown cells.

~~~bash
lispyn build examples/hello.hy -o hello.ipynb
~~~

**3. Build and execute a notebook.** Jupyter runs the cells; `hello-executed.ipynb` saves the executed cells and their outputs.

~~~bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
~~~

The hello example prints `2 + 3 + 5 = 10` and `Lispyn checks passed!`. For real but tiny offline PyTorch training, try `lispyn execute examples/quickstart.hy` after installing torch.

## When should you use it?

Choose Lispyn when you like Lisp/Hy macros, want **plain Python notebook outputs**, and repeatedly build comparable ML experiments with dataset checks, structured reporting and reusable workflows. Stay with regular Python notebooks when you do not need macros, or prefer editing cells interactively as your primary source. Lispyn does not make models more accurate, guarantee bitwise determinism, or prevent data leakage unless the experiment is designed correctly.

**Security:** Only compile and execute trusted Hy code: Hy macros can execute arbitrary Python at compile time. Build does not run notebook cells, but compilation itself is **not sandboxed**. Load checkpoints only from trusted sources.

## Documentation

[**Complete reference: 72 supported macros**](supported_macros/README.md) — arguments, dependencies and runnable Hy examples for PyTorch, NumPy, pandas, Matplotlib, Jupyter authoring and experiment analysis. Also see [generated notebook examples](example_generated_notebooks/README.md), [changelog](CHANGELOG.md) and [citation](CITATION.cff).

*Created by Eduardo J. Barrios · Apache-2.0 licensed · experimental alpha software.*
