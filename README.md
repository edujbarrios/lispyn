# Lispyn

**Write research experiments in Hy (Lisp for Python), then run them directly or turn them into standard Jupyter notebooks.**

Lispyn adds reusable macros to existing **PyTorch, Matplotlib, pandas and NumPy** workflows. It does not replace those libraries or lock you into a training framework.

[![PyPI version](https://img.shields.io/pypi/v/lispyn?label=PyPI%20version)](https://pypi.org/project/lispyn/) [![CI on main](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/lispyn/actions/workflows/tests.yml?query=branch%3Amain) [![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why use it for research?

- **Less boilerplate:** repeatable experiment parameters, training, data checks, splits, cross-validation, metrics and plotting.
- **One source, three modes:** execute Hy directly; compile notebook cells without running them; or run and save the notebook with outputs.
- **Inspectable research artifacts:** ordinary Python cells, provenance, reports and optional integrity-checked experiment archives.

These tools help organize research; they do not guarantee reproducibility or scientifically valid results.

## Install

```bash
pip install lispyn
```

Hy, notebook execution and Matplotlib are included. NumPy comes with Matplotlib; for pandas use `pip install "lispyn[data]"`. **PyTorch is installed separately** for the appropriate [CPU/GPU build](https://pytorch.org/get-started/locally/).

## How it works: three ways to run Hy

All three commands use the same [runnable `examples/hello.hy`](examples/hello.hy). Clone the examples first:

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
```

**1. Run Hy directly** — prints to the terminal, no notebook:

```bash
lispyn run examples/hello.hy
```

**2. Build a Jupyter notebook** — creates `hello.ipynb` with Python/Markdown cells. **Nothing is executed**:

```bash
lispyn build examples/hello.hy -o hello.ipynb
```

**3. Build and execute a Jupyter notebook** — creates `hello-executed.ipynb` and **saves the executed cells** and their outputs:

```bash
lispyn execute examples/hello.hy -o hello-executed.ipynb
```

Check for `2 + 3 + 5 = 10` and `Lispyn checks passed!` in the terminal or executed notebook. Hy macros compile into regular Python notebook cells. Only run or compile **trusted Hy source**; macros are not sandboxed.

## Complete macro catalog

**[All 49 macros: syntax, arguments, dependencies and Hy examples](supported_macros/README.md)**

The catalog explains the available **PyTorch** training and evaluation, **Matplotlib** charts, **pandas** table operations, **NumPy** array checks, notebook generation, reports and experiment comparisons. New: [data-quality profiles](examples/data_profiles.hy). See also the [PyTorch quickstart](examples/quickstart.hy) and [executed notebook gallery](example_generated_notebooks/).

For native PyTorch, try `lispyn execute examples/quickstart.hy` after installing PyTorch.

From a source checkout, `pip install -r requirements.txt` installs an optional full research environment.

*Created by Eduardo J. Barrios · [Apache-2.0](LICENSE) · [Citation](CITATION.cff) · Experimental alpha software.*
