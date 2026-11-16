# Lispyn

**A new way to write and run machine-learning experiments.**

*Lisp for expressive experiments · Python for the ecosystem · Jupyter for sharing results.*

[![PyPI](https://img.shields.io/pypi/v/lispyn?label=PyPI)](https://pypi.org/project/lispyn/)
[![Tests](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/lispyn)](https://pypi.org/project/lispyn/)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2563eb.svg)](LICENSE)

**Go from experiment idea to reproducible results in one Hy/Lisp file.** Write expressive experiments using Lispyn macros, run your existing PyTorch models, and publish the result as an ordinary, executable Jupyter notebook—complete with parameters, explanations, training curves and metrics. **Keep the Python ecosystem you already know.**

Lispyn adds a new **way to author experiments**, not a new training framework. Your data, models and optimizers stay native PyTorch.

[**Explore all 43 supported macros**](supported_macros/README.md) · [Hy source](examples/quickstart.hy) · [Executed notebook](example_generated_notebooks/quickstart_executed.ipynb) · [Example gallery](example_generated_notebooks/)

## Install

Python 3.10+ is supported. **One command gives you Lispyn, Jupyter notebook execution and Matplotlib plotting:**

```bash
pip install lispyn
```

No `[notebook,plots]` extras needed. Run `lispyn doctor` to check your environment. **PyTorch is deliberately separate**: it is large and its CPU/CUDA/ROCm build depends on your system. For PyTorch experiments, install the [appropriate official build](https://pytorch.org/get-started/locally/) (on a compatible CPU environment, `pip install torch` is a convenient starting point).

**Full research environment from a clone** (Lispyn, PyTorch, Matplotlib, Jupyter, NumPy and pandas):

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
pip install -r requirements.txt
lispyn doctor
```

The root [`requirements.txt`](requirements.txt) is an optional research setup, not a lockfile. The `notebook` and `plots` extras remain available for compatibility but their libraries now install by default; `data`, `vision`, `audio` and `peft` cover additional specialized workflows.

### Need another library inside a notebook?

Use Lispyn's **`pip-install` macro** directly in your Hy experiment:

```hy
(require lispyn.macros [pip-install])
(pip-install "scikit-learn>=1.5")
```

When Lispyn compiles the `.hy` source, this becomes a dedicated dependency cell in the standard Jupyter notebook. **`lispyn build` does not install anything; `lispyn execute` checks and installs the named package only if needed**, using the notebook's Python environment. It accepts ordinary named requirements, not arbitrary pip shell commands or package URLs. Execute only trusted notebooks; a kernel restart may be needed after adding native libraries. For GPU-specific PyTorch versions, use the official installation instructions rather than this macro.

## From a Lisp experiment to the finished notebook

This is **actual PyTorch training**, not a fabricated plot or an estimated metric. The model trains on deliberately small **synthetic classification data** (CPU only, no downloads) and reserves a held-out test subset. The numbers are illustrative, not scientific performance claims.

Below is a **complete, runnable notebook example in Hy**, followed immediately by its **actual recorded metrics and training graph**. You can also open the generated `.ipynb` to inspect all the executed Python/Jupyter cells.

<!-- QUICKSTART_EXECUTED:START -->
### A complete notebook written in Hy/Lispyn

This is the **complete, runnable source** of [`examples/quickstart.hy`](examples/quickstart.hy). The `md` macros become notebook Markdown cells; the remaining forms become ordinary Python code cells.

```hy
; Real CPU PyTorch training on deliberately small SYNTHETIC data.
; Every notebook cell and the final figure are recorded, not fabricated.
(require lispyn.macros [md])
(md "# Quickstart · from Lisp to an executed research notebook"
    "A native PyTorch classifier on synthetic data. We reserve an unseen"
    "test partition for a final, illustrative evaluation.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 10))

(require lispyn.macros [md])
(md "## 1. Prepare data and isolate the test set"
    "Create two-dimensional toy samples; split them before training.")

(require lispyn.macros [train-val-test])
(import torch)
(torch.manual_seed seed)
(setv features
  (torch.tensor
    [[0.0 0.0] [0.0 1.0] [1.0 0.0] [1.0 1.0]
     [0.2 0.3] [0.3 0.9] [0.9 0.3] [0.8 0.9]
     [0.1 0.2] [0.1 0.8] [0.9 0.2] [0.9 0.8]
     [0.4 0.1] [0.4 0.9] [0.7 0.1] [0.7 0.9]]))
(setv labels (torch.tensor [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]))
(setv dataset (torch.utils.data.TensorDataset features labels))
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.25
    :test-fraction 0.25 :seed seed))
(print "Samples:" (len training) "train," (len validation)
  "validation," (len held-out) "held-out test")

(require lispyn.macros [md])
(md "## 2. Train a native PyTorch model"
    "Lispyn handles the training loop and early stopping; model, optimizer"
    "and tensors remain ordinary PyTorch objects.")

(require lispyn.macros [train-until])
(setv model (torch.nn.Linear 2 2))
(setv run
  (train-until model training 3
    :validation validation :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs epochs :batch-size 4 :lr 0.05 :seed seed :device "cpu"))
(print "Completed epochs:" (len (get run.history "train_loss")))

(require lispyn.macros [md])
(md "## 3. Report and evaluate"
    "Summarize recorded metrics, then evaluate the untouched test subset once."
    "This tiny synthetic dataset is a workflow demo, not research evidence.")

(require lispyn.macros [report-run narrate metric-table])
(import lispyn.training [evaluate-metrics])
(report-run run :title "PyTorch baseline · recorded metrics")
(setv held-out-loader (torch.utils.data.DataLoader held-out :batch_size 4))
(setv test-scores
  (evaluate-metrics run.model held-out-loader
    :loss-fn (torch.nn.CrossEntropyLoss) :metrics "accuracy" :device "cpu"))
(narrate "Held-out test accuracy: **{accuracy:.1%}**"
  :accuracy (get test-scores "accuracy"))
(metric-table
  {"best_val_loss" (min (get run.history "val_loss"))
   "test_loss" (get test-scores "loss")
   "test_accuracy" (get test-scores "accuracy")}
  :title "Recorded experiment scores" :precision 4)

(require lispyn.macros [md])
(md "## 4. Plot the real training history"
    "Generated from the run.history actually recorded above, never fake curves.")

(require lispyn.macros [curve])
(curve run :metric "loss" :theme "paper"
  :title "Quickstart · real PyTorch training"
  :subtitle "Train versus validation loss · synthetic dataset"
  :figsize [8 4.5]
  :save "../example_generated_notebooks/assets/quickstart_training_curves.png")
```

### Results from the executed notebook

These values come directly from the [**executed Jupyter notebook**](example_generated_notebooks/quickstart_executed.ipynb), not from manually written or simulated results.

**Dataset and execution**

```text
Samples: 8 train, 4 validation, 4 held-out test
Completed epochs: 10
```

**Training report**

# PyTorch baseline · recorded metrics

**Epochs recorded:** 10
**Early stopping:** no

## Metrics

| Metric | Optimize | Best epoch | Best value | Final value |
| --- | --- | ---: | ---: | ---: |
| train_loss | min | 10 | 0.440579 | 0.440579 |
| val_loss | min | 10 | 0.429638 | 0.429638 |

_Best epochs are 1-based. This report contains only metric summaries, not raw dataset rows, predictions, or model weights._

**Held-out evaluation**

Held-out test accuracy: **50.0%**

### Recorded experiment scores

| Metric | Value |
| --- | ---: |
| best_val_loss | 0.4296 |
| test_loss | 0.7664 |
| test_accuracy | 0.5 |

### Final graph — training vs. validation loss

![Loss curves generated by Lispyn from real quickstart training](example_generated_notebooks/assets/quickstart_training_curves.png)

**[Open the full executed notebook](example_generated_notebooks/quickstart_executed.ipynb)** · [View original Hy source](examples/quickstart.hy)

**Verified Hy source SHA-256:** `1712e572ac12fde0249c8fc45efcd222cc586b40b8b580bbea1680e1cb6384ea`.

*The data are deliberately synthetic and small: this demonstrates the workflow, not model performance on a real research task.*
<!-- QUICKSTART_EXECUTED:END -->

## Reproduce the result

Run the exact Hy file shown above:

```bash
# Build only: Hy becomes an inspectable Python notebook; no training.
lispyn build examples/quickstart.hy

# Actually train on CPU and capture *every* notebook cell and figure.
lispyn execute examples/quickstart.hy \
  -o example_generated_notebooks/quickstart_executed.ipynb

# Verify the source fingerprint and that code cells executed without errors.
lispyn audit example_generated_notebooks/quickstart_executed.ipynb \
  --source examples/quickstart.hy

# Export the notebook as a portable Markdown report.
lispyn export example_generated_notebooks/quickstart_executed.ipynb \
  -o quickstart_report.md
```

The plot is written to `example_generated_notebooks/assets/quickstart_training_curves.png` by the `curve` **Hy macro**, from the actual training history. To change an experiment parameter without editing the source, use `lispyn build examples/quickstart.hy --set epochs=20`.

## Discover your best experiments automatically

Save runs with `:experiment-dir "runs/baseline"` from the native `train-until` macro, then compare all saved histories **without manually listing each run**:

```bash
lispyn leaderboard runs/ --metric loss --rank-by best -o leaderboard.md
```

Or write the comparison directly inside your Hy notebook:

```hy
(require lispyn.macros [experiment-leaderboard])
(experiment-leaderboard "runs" :metric "accuracy" :mode "max"
  :save "runs/leaderboard.md")
```

Lispyn discovers completed run histories, ranks their **best or final** scores and shows a shareable Markdown table. It skips incomplete or incompatible runs by default (`--strict` turns that into an error), and supports CSV/JSON export. It reads **only saved aggregate metrics**, not checkpoints or training data. Rankings are descriptive, not proof of statistical significance.

Explore the [synthetic Hy leaderboard example](examples/experiment_leaderboard.hy) or see the [macro catalog](supported_macros/README.md).

## Keep experimenting in Hy

You can use the same approach with early stopping, grouped and temporal cross-validation, parameter sweeps, model-head fine-tuning, confusion matrices, evidence checks and reusable chart styles. The [**complete macro reference**](supported_macros/README.md) documents all **43** public macros with Hy examples.

Explore additional examples in [`examples/`](examples/) and their compiled [Jupyter notebooks](example_generated_notebooks/). Lispyn is an **alpha** project; interfaces may change as it evolves.

**For contributors:** add a `defmacro` in [`src/lispyn/macros.hy`](src/lispyn/macros.hy), update [`supported_macros/README.md`](supported_macros/README.md) and include a runnable example. CI checks the catalog against the public macros.

---

Created by **Eduardo J. Barrios** · [Apache-2.0](LICENSE) · [GitHub issues](https://github.com/edujbarrios/lispyn/issues)
