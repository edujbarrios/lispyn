# Lispyn — Lisp-powered AI notebooks

**Write AI experiments as Hy macros. Run them directly, compile them to readable Python, or execute them as Jupyter notebooks.** Lispyn helps you define native PyTorch models, train and evaluate them, create Matplotlib figures, validate datasets and publish reproducible results. **No new tensor framework or opaque training runtime.**

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use Lispyn for AI notebooks?

**Look at what you can write.** This is real Hy syntax from the [runnable CPU showcase](examples/showcase.hy), not pseudocode. The macros generate native Python functions and Matplotlib calls.

### 1. Declare a dependency cell in your notebook

```hy
(require lispyn.macros [pip-install])
(pip-install "matplotlib>=3.8")
```

`pip-install` produces an isolated Jupyter dependency cell. **Nothing is executed during `lispyn build`**; the packages are checked/installed only when you run the Hy script or execute the notebook. Matplotlib is already in the default Lispyn package. Install the appropriate PyTorch CPU/GPU build **separately**—don't assume `pip-install "torch"` selects the right accelerator.

### 2. Create a real PyTorch model, optimizer and training loop

```hy
(require lispyn.macros [defmodel defoptimizer deftrain-step collect-epochs])
(import torch)
(torch.manual_seed 42)

(defmodel make-mlp [features hidden]
  (torch.nn.Linear features hidden)
  (torch.nn.ReLU)
  (torch.nn.Linear hidden 1))
(setv model (make-mlp 2 8))
(defoptimizer make-adam [model rate] torch.optim.Adam :weight-decay 0.01)
(setv optimizer (make-adam model 0.05))
```

`defmodel` creates a new `torch.nn.Sequential` model each call; `defoptimizer` creates a standard `torch.optim.Adam`. Compose them with `deftrain-step`, `collect-epochs`, `with-training` and `with-eval`. For custom architectures, use ordinary `torch.nn.Module`.

### 3. Make an original plot with a reusable macro

```hy
(require lispyn.macros [defplot with-figure])
(import matplotlib.pyplot :as plt)

(defplot loss-chart [losses] [fig ax]
  (.plot ax (range (len losses)) losses :label "Training loss" :marker "o")
  (.set_title ax "Training progress")
  (.set_xlabel ax "Epoch")
  (.set_ylabel ax "MSE")
  (.legend ax))
(setv figure (loss-chart losses))
(.savefig figure "showcase-loss.png")
(plt.close figure)
```

`defplot` generates a reusable Python function that returns a Matplotlib figure. `with-figure` builds a one-off chart with access to the actual `fig` and `ax` objects. For recorded training metrics, `curve` and `defcurve` provide prepared themes, comparison plots and PNG/SVG exports.

**Try the complete example:** `lispyn run examples/showcase.hy`, or use `build` and `execute` to create a shareable notebook. The example trains on four synthetic CPU samples, prints losses, and saves two real PNG figures. It does not claim meaningful predictive performance.

## What else can Lispyn do?

| Your goal | Hy macros and examples |
| --- | --- |
| **Notebook composition** | `md`, `pip-install`, `experiment-params` — [notebook cells](examples/markdown_cells.hy) |
| **PyTorch control** | `with-seed`, `with-autocast`, `clip-gradients`, `save-model`, `load-model` — [CPU controls](examples/pytorch_controls.hy) |
| **Data quality and safe evaluation** | `assert-shape`, `assert-finite`, `assert-disjoint`, `assert-probabilities` — [evaluation guards](examples/evaluation_guards.hy) |
| **Validation and research** | `train-val-test`, `cross-validate`, `train-until`, `class-weights` — [real quickstart](examples/quickstart.hy) |
| **Plots and reports** | `curve`, `defcurve`, `confusion-matrix`, `report-run`, `metric-table` — [curve designs](examples/curve_designs.hy) |
| **Inspect and extend the language** | `deftrain-step`, `collect-epochs`, `defmodel`, `defplot` — [macro authoring](MACRO_AUTHORING.md) |

All **84 supported macros** have [Hy examples in the reference](supported_macros/README.md). The [executed quickstart notebook](example_generated_notebooks/quickstart_executed.ipynb) shows real PyTorch results.

## Install

```bash
pip install lispyn
```

Python 3.10+ is required. The base install includes Hy, notebook compilation/execution, NumPy via Matplotlib, and Matplotlib itself; pandas is optional (`pip install "lispyn[data]"`). **PyTorch is installed separately** using [PyTorch's official platform-specific instructions](https://pytorch.org/get-started/locally/).

For the checked-in examples, clone the repository:

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
# Optional, larger research environment:
pip install -r requirements.txt
```

## One Hy source → three workflows

The tiny [hello example](examples/hello.hy) requires no PyTorch:

```bash
lispyn run examples/hello.hy
lispyn build examples/hello.hy -o hello.ipynb
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

- **`run`:** run Hy directly, without creating a notebook.
- **`build`:** compile unexecuted, inspectable Python and Markdown cells; **Nothing is executed** at runtime.
- **`execute`:** run Jupyter and save the executed cells and real outputs.

Explore expansion with `hy2py examples/showcase.hy`. Use `lispyn new study.hy` to scaffold a notebook or `lispyn gallery` for a set of compiled examples.

## Where Lispyn fits

Lispyn is **alpha software**: macro interfaces may evolve. Choose it if source transformations, reusable experiment scaffolding and auditable Python notebooks are valuable to you; stick with plain Python when macros don't help. Generated Python still depends on your original libraries and data.

**Security and limits:** Hy macros execute Python at compile time. Compile only trusted `.hy` sources, including during `build`; it is not sandboxed. Notebook `pip-install` runs only when executed, and package installation requires trusted dependencies. No macro guarantees determinism, absence of leakage, or improved model accuracy.

[**Macro catalog**](supported_macros/README.md) · [**Macro-authoring guide**](MACRO_AUTHORING.md) · [**Notebook gallery**](example_generated_notebooks/README.md) · [**Changelog**](CHANGELOG.md) · [**Citation**](CITATION.cff)

*Created by Eduardo J. Barrios · Apache-2.0 licensed.*
