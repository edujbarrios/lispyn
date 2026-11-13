# Lispyn

**Lisp for experiments. Jupyter for results.**

**Write expressive machine-learning experiments in Lisp, turn repetitive steps into reusable macros, and generate executable Jupyter notebooks with Python.** Lispyn keeps your experiments concise and easy to adapt while giving you direct access to PyTorch, TorchVision, TorchAudio and the Python ecosystem.

## Why Lisp?

Jupyter is great for exploring results; Lisp is great for **expressing and reusing the steps that produce them**. Its code-as-data design lets you turn recurring experiment patterns into small macros instead of copying cells between notebooks. Lispyn builds on [Hy](https://hylang.org/) to translate that Lisp source into Python notebooks you can inspect, run and share.

**What it is:** Hy syntax and macros, plus focused helpers for datasets, training and Jupyter. **What it isn't:** a new interpreter, a replacement for PyTorch, or a mature standalone language. The macros are focused (`with-seed`, `with-eval`, `curve`, `defcurve`, `train-for`); most current features are ordinary Python functions callable from Hy. We're building the abstractions incrementally.

## A short experiment

```clojure
(import torch)
(import lispyn.workflow [train])
(require lispyn.macros [with-seed])

(with-seed 42
  (setv model (torch.nn.Linear 2 2)))

(setv data (torch.utils.data.TensorDataset
  (torch.tensor [[0.0 0.0] [1.0 0.0]])
  (torch.tensor [0 1])))

(setv run
  (train model data
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 3
    :seed 42
    :experiment-dir "runs/demo"))
```

This runs real PyTorch. The optional experiment directory stores `manifest.json` (seed, configuration, dependency versions and best score), `history.json` and `best.pt`. The `.hy` source remains the editable source of truth.

## Install and run

Requires Python 3.10+. Install **PyTorch for your CPU/GPU** from the [official instructions](https://pytorch.org/get-started/locally/) first.

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
python -m pip install -e ".[notebook,data]"

# Start a new Lisp experiment
lispyn new experiment.hy

# Hy → inspectable Python Jupyter notebook (no cell execution)
lispyn build examples/reproducible_run.hy

# Generate + execute notebook, preserving results
lispyn execute examples/reproducible_run.hy

# Read the recorded experiment
lispyn report examples/runs/smoke

# Optional: save a training curve (pip install -e ".[plots]")
lispyn report examples/runs/smoke --plot examples/runs/smoke/loss.png

# Run Hy directly, without a notebook
lispyn run examples/first_experiment.hy
```

`new` creates an editable Hy experiment without running it or overwriting an existing file. `build` compiles Hy into a `.ipynb` with Python code cells; `execute` runs that notebook; `run` uses Hy directly. Markers such as `; %% [markdown]` create separate notebook sections.

## Design curves once, reuse them

Create a personal Matplotlib style as a Lisp macro, then apply it to any recorded training run:

```clojure
(require lispyn.macros [defcurve])
(defcurve my-style :theme "dark" :train-color "#22D3EE" :val-color "#FBBF24")
(my-style result :metric "loss" :title "Training loss" :save "loss.png")
(my-style result :metric "accuracy" :title "Accuracy" :save "accuracy.png")
```

Also available: `(curve result :theme "paper" :title "...")`. Choose colors,
labels, line widths, figure dimensions and export paths. Install optional
`.[plots]` support; see [the reusable curve example](examples/curve_designs.hy).
## What works today

| Area | Current support |
| --- | --- |
| **Data** | CSV/JSON/Parquet, reproducible and stratified PyTorch dataset splits |
| **Images** | TorchVision image folders, transforms, pretrained-model recipes |
| **Text** | Text files, tokenization collators, Transformers and PEFT LoRA |
| **Audio** | TorchAudio decoding, labeled folders and spectrograms |
| **Training** | Head-only/full fine-tuning, LoRA, accuracy, checkpoints, metrics |
| **Reproducibility** | Seeds, recorded environment/configuration, run reports |
| **Jupyter** | Generate and execute notebooks; reusable Matplotlib curve designs |

Examples: [`plot_experiment.hy`](examples/plot_experiment.hy), [`stratified_inference.hy`](examples/stratified_inference.hy), [`one_call_training.hy`](examples/one_call_training.hy), [`one_call_lora.hy`](examples/one_call_lora.hy), [`vision_finetune.hy`](examples/vision_finetune.hy), [`audio_features.hy`](examples/audio_features.hy). Optional integrations need their compatible libraries installed.

## Direction

1. **Keep source concise:** useful Lisp macros, no imitation of the PyTorch API.
2. **Keep experiments inspectable:** generated Python notebooks and explicit outputs.
3. **Keep the ecosystem open:** native Python functions, community macros, small tested releases.

**Experimental v0.10.0-alpha:** APIs may change. Recording a seed and package versions helps repeat an experiment but does not guarantee deterministic GPU execution. Notebook compilation may execute Hy macros: **run only trusted `.hy` files**. The run manifest does not store dataset contents or secrets; using the same data and base weights remains your responsibility.

For API details, optional installation and advanced examples, see the [extended guide](docs/guide.md). Contributions and issues are welcome.

## Author and license

Created by **Eduardo J. Barrios**. Copyright © 2026 Eduardo J. Barrios. Licensed under **Apache-2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Redistribution must preserve applicable notices. The license does not require attribution merely for executing a program. Generated notebooks include author attribution by default.
