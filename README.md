# Lispyn — Realización de notebooks de IA

**Create AI notebooks with Hy (Lisp for Python): run scripts, generate Jupyter notebooks or save executed results.**

Lispyn provides macros for **PyTorch, NumPy, pandas and Matplotlib** while keeping your Python libraries and models.

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use Lispyn for AI notebooks?

**Less boilerplate.** For example, [the real PyTorch quickstart](examples/quickstart.hy) uses a macro to create separate train, validation and held-out test subsets:

```hy
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.25
    :test-fraction 0.25 :seed seed))
```

**Catch invalid data early.** [The runnable NumPy/pandas example](examples/validate_inputs.hy) checks shape and finite values without writing a manual validation loop:

```hy
(require lispyn.macros [assert-finite assert-shape])
(import numpy :as np)
(import pandas :as pd)

(setv features (np.array [[1.0 2.0] [3.0 4.0]]))
(assert-shape features [2 2] :name "features")
(assert-finite features :name "features")
```

**Check class imbalance before training.** The [runnable label balance example](examples/label_balance.hy) summarizes a classification dataset and plots its class counts:

```hy
(require lispyn.macros [label-balance plot-label-balance])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv report (label-balance labels :title "Training label distribution"))
```

**One source, three modes.** The same [hello.hy](examples/hello.hy) can run in a terminal, become an unexecuted notebook, or produce a notebook with saved outputs. See the three commands below.

**Inspectable AI notebook outputs.** The [quickstart](examples/quickstart.hy) renders a report from actual PyTorch training results, which can be reviewed in the [executed notebook](example_generated_notebooks/quickstart_executed.ipynb):

```hy
(require lispyn.macros [report-run narrate metric-table])
(import lispyn.training [evaluate-metrics])
(report-run run :title "PyTorch baseline · recorded metrics")
```

These examples simplify AI notebook development; Lispyn does not guarantee model quality, reproducibility or valid evaluations.

## Install

```bash
pip install lispyn
```

Hy, Jupyter execution and Matplotlib are included. NumPy is available through Matplotlib; for pandas install `pip install "lispyn[data]"`. **PyTorch is installed separately** for your [CPU/GPU setup](https://pytorch.org/get-started/locally/).

## How it works: three ways to run Hy

Get the [hello.hy example](examples/hello.hy) (no dataset, no PyTorch):

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
```

**1. Run the Hy file** — terminal output, no notebook:

```bash
lispyn run examples/hello.hy
```

**2. Generate a Jupyter notebook** — `hello.ipynb`, **Nothing is executed**:

```bash
lispyn build examples/hello.hy -o hello.ipynb
```

**3. Generate and execute the notebook** — `hello-executed.ipynb` **saves the executed cells** and outputs:

```bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

Check for `2 + 3 + 5 = 10` and `Lispyn checks passed!`. Building compiles Hy into ordinary Python/Markdown notebook cells without executing it. **Only compile trusted Hy:** macros are not sandboxed.

## All macros and examples

**[Full reference: 53 macros, arguments, dependencies and Hy examples](supported_macros/README.md)** — grouped by PyTorch, Matplotlib, pandas, NumPy, notebook authoring, classification and AI experiment reporting.

More runnable sources: [PyTorch quickstart](examples/quickstart.hy) (try `lispyn execute examples/quickstart.hy`), [data profiles](examples/data_profiles.hy) and [input assertions](examples/validate_inputs.hy) and [label balance](examples/label_balance.hy). A cloned repository also offers `pip install -r requirements.txt` for an optional full AI development environment.

*Created by Eduardo J. Barrios · [Apache-2.0](LICENSE) · [Citation](CITATION.cff) · Experimental alpha software.*
