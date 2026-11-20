# Lispyn — Lisp-powered AI notebooks

**Lispyn is a macro-first toolkit for authoring AI experiments in Hy (Lisp for Python).** Define and compose reusable **compile-time macros**, then run your experiments directly or compile them into ordinary Python Jupyter notebooks.

The core product is a **language of Hy macros**, not a replacement for PyTorch, NumPy, pandas, Matplotlib or Python. Macro expansion makes experiment structure explicit and inspectable; small Python helpers handle numerical operations where appropriate. Your models, data, optimizers and evaluation logic remain native Python.

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use Lispyn for AI notebooks?

**Macros are the main abstraction.** Hy macros transform source code at compilation, rather than just exposing another training API. Compose native Python operations, build reusable training functions, and create your own macros. Version 0.39 introduces `deftrain-step`, `optimizer-step`, `collect-epochs`, `with-training`, `with-grad` and `with-rng-fork` — six self-contained macros that expand into Python without a new Lispyn runtime backend.

```hy
(require lispyn.macros [deftrain-step collect-epochs])
(deftrain-step train-one model optimizer criterion)
(setv history (collect-epochs epoch 4
  (train-one features labels)))
```

See the [runnable CPU macro-first example](examples/macro_first_training.hy) and the [guide to developing your own macros](MACRO_AUTHORING.md), including hygienic names, composition, independent notebook cells and compile-time security.

**One source, three artifacts.** Write a `.hy` experiment once. Run it as a script (`run`), compile an unexecuted notebook (`build`), or execute it and save real outputs (`execute`). The resulting notebook contains ordinary Python, not an opaque Lispyn engine.

**Data splitting and explicit checks.** The [real PyTorch quickstart](examples/quickstart.hy) splits data into training, validation and held-out test sets:

```hy
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.25
    :test-fraction 0.25 :seed seed))
```

The [input validation example](examples/validate_inputs.hy) rejects bad shapes and nonfinite numbers:

```hy
(require lispyn.macros [assert-finite assert-shape])
(import numpy :as np)
(import pandas :as pd)

(setv features (np.array [[1.0 2.0] [3.0 4.0]]))
(assert-shape features [2 2] :name "features")
(assert-finite features :name "features")
```

Check classification class balance with the [runnable example](examples/label_balance.hy):

```hy
(require lispyn.macros [label-balance plot-label-balance])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv report (label-balance labels :title "Training label distribution"))
```

For classification, `class-weights` computes inverse-frequency loss weights, and `threshold-report` compares validation probability cutoffs without peeking at held-out test labels.

**Native PyTorch, not a parallel framework.** Use macros such as `with-eval`, `with-training`, `with-autocast`, `clip-gradients`, `with-parameter-freeze`, `gradient-report` and `predict-batches` to generate inspectable operations or invoke explicit low-level checks. Training, autograd, optimizers, and saved checkpoints remain standard PyTorch. The [PyTorch controls](examples/pytorch_controls.hy) and [diagnostics example](examples/torch_diagnostics.hy) run on the CPU.

**Evidence-driven notebooks.** The [quickstart](examples/quickstart.hy) reports real PyTorch metrics:

```hy
(require lispyn.macros [report-run narrate metric-table])
(import lispyn.training [evaluate-metrics])
(report-run run :title "PyTorch baseline · recorded metrics")
```

Generate plots, comparisons, parameterized notebooks, and reports from run data. See the [executed quickstart notebook](example_generated_notebooks/quickstart_executed.ipynb).

## Install

```bash
pip install lispyn
```

Python 3.10+ is required. The base package includes Hy, notebook compilation/execution, and Matplotlib. Install pandas with `pip install "lispyn[data]"`. **PyTorch is installed separately** using the appropriate CPU or GPU build from [PyTorch's installation guide](https://pytorch.org/get-started/locally/). A cloned repository also offers the broader research environment:

```bash
pip install -r requirements.txt
```

Lispyn is experimental alpha software; macro interfaces may evolve.

## Quickstart: run, build, execute

Clone the repository to access examples (the hello example does not require PyTorch):

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
```

**1. Run a Hy experiment.** Nothing is written to a notebook:

```bash
lispyn run examples/hello.hy
```

**2. Build a shareable notebook.** Nothing is executed; compilation creates ordinary Python and Markdown cells:

```bash
lispyn build examples/hello.hy -o hello.ipynb
```

**3. Execute and capture outputs.** Jupyter runs the compiled notebook and saves the executed cells:

```bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

The hello example prints `2 + 3 + 5 = 10` and `Lispyn checks passed!`. For offline CPU training, run `lispyn execute examples/macro_first_training.hy` after installing torch. To inspect macro expansion directly, use `hy2py examples/macro_first_training.hy`.

## Design principles and limits

Choose Lispyn when **Lisp/Hy macros are worth the cognitive overhead**, you want versioned experiments as source files, or you need reviewable standard-Python Jupyter artifacts. Use plain Python when interactive notebook editing is your main workflow or you do not need source transformations.

**Macro-first means:** focus on hygienic syntax transformations, scoped state management, and composable experiment code. It does **not** mean every numerical algorithm belongs inside a macro. Python backends are appropriate for substantial calculations and validations; macros provide the public authoring interface.

**Security:** Compile and execute only trusted Hy code. Hy macros can execute arbitrary Python at **compile time**, so even `lispyn build` is not sandboxed. Load checkpoints only from trusted sources. Lispyn cannot guarantee bitwise determinism, better accuracy, or absence of data leakage.

## Documentation

[**Reference: 78 supported macros**](supported_macros/README.md) · [**Writing and extending macros**](MACRO_AUTHORING.md) · [CPU macro-first walkthrough](examples/macro_first_training.hy) · [Notebook gallery](example_generated_notebooks/README.md) · [Changelog](CHANGELOG.md) · [Citation](CITATION.cff)

*Created by Eduardo J. Barrios · Apache-2.0 licensed · experimental alpha software.*
