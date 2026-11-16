# Lispyn

**Write experiments in Lisp. Train with PyTorch. Share real, executable Jupyter notebooks.**

*Lisp for expressive experiments · Python for the ecosystem · Jupyter for sharing results.*

[![PyPI](https://img.shields.io/pypi/v/lispyn?label=PyPI)](https://pypi.org/project/lispyn/)
[![Tests](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/lispyn)](https://pypi.org/project/lispyn/)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2563eb.svg)](LICENSE)

Lispyn gives you **Hy/Lisp macros** for reproducible, native PyTorch experiments. Instead of maintaining a script *and* manually copying results into a notebook, you write one `.hy` experiment and compile or execute it as a **standard Python/Jupyter notebook**—with parameter cells, explanations, training histories and plots.

**No replacement machine-learning framework.** Your PyTorch models, data loaders, tensors and optimizers are still PyTorch.

[**Explore all 42 supported macros**](supported_macros/README.md) · [Hy source](examples/quickstart.hy) · [Executed notebook](example_generated_notebooks/quickstart_executed.ipynb) · [Example gallery](example_generated_notebooks/)

## Install

Requires Python 3.10+. Use a virtual environment if possible.

```bash
python -m pip install --upgrade "lispyn[notebook,plots]"
python -m pip install torch
lispyn doctor
```

For a CUDA/ROCm or platform-specific PyTorch build, follow [the official installation guide](https://pytorch.org/get-started/locally/). Lispyn supports optional extras for pandas, vision, audio and PEFT; the example below needs only PyTorch, Jupyter and Matplotlib.

## From a Lisp experiment to the finished notebook

This is **actual PyTorch training**, not a fabricated plot or an estimated metric. The model trains on deliberately small **synthetic classification data** (CPU only, no downloads) and reserves a held-out test subset. The numbers are illustrative, not scientific performance claims.

The same `examples/quickstart.hy` creates each notebook cell, including the final Matplotlib training curve. The recorded outputs are extracted **directly from the executed `.ipynb`** and published below.

<!-- QUICKSTART_EXECUTED:START -->
_The complete executed-cell snapshot is generated automatically from the saved notebook by `scripts/render_readme_quickstart.py`. It appears here after the validated example runs in CI._
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

## Keep experimenting in Hy

You can use the same approach with early stopping, grouped and temporal cross-validation, parameter sweeps, model-head fine-tuning, confusion matrices, evidence checks and reusable chart styles. The [**complete macro reference**](supported_macros/README.md) documents all **42** public macros with Hy examples.

Explore additional examples in [`examples/`](examples/) and their compiled [Jupyter notebooks](example_generated_notebooks/). Lispyn is an **alpha** project; interfaces may change as it evolves.

**For contributors:** add a `defmacro` in [`src/lispyn/macros.hy`](src/lispyn/macros.hy), update [`supported_macros/README.md`](supported_macros/README.md) and include a runnable example. CI checks the catalog against the public macros.

---

Created by **Eduardo J. Barrios** · [Apache-2.0](LICENSE) · [GitHub issues](https://github.com/edujbarrios/lispyn/issues)
