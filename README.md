# Lispyn

**Lisp for experiments. Jupyter for results.**

**Write expressive machine-learning experiments in Lisp, turn repetitive steps into reusable macros, and generate executable Jupyter notebooks with Python.** Lispyn keeps your experiments concise and easy to adapt while giving you direct access to PyTorch, TorchVision, TorchAudio and the Python ecosystem.

## Why Lisp?

Jupyter is great for exploring results; Lisp is great for **expressing and reusing the steps that produce them**. Its code-as-data design lets you turn recurring experiment patterns into small macros instead of copying cells between notebooks. Lispyn builds on [Hy](https://hylang.org/) to translate that Lisp source into Python notebooks you can inspect, run and share.

**What it is:** Hy syntax and macros, plus focused helpers for datasets, training and Jupyter. **What it isn't:** a new interpreter, a replacement for PyTorch, or a mature standalone language. The macros are focused (`md`, `head-tune`, `full-tune`, `with-seed`, `with-eval`, `curve`, `defcurve`); most current features are ordinary Python functions callable from Hy. We're building the abstractions incrementally.

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
## Markdown cells from Lisp macros

Write notebook documentation *in the same Lisp source* as your code. Top-level
`md` or `markdown` forms containing literal text become **real Jupyter Markdown cells**:

```clojure
(require lispyn.macros [md])
(md "## Load the dataset" "A short explanation in **Markdown**.")
(import torch)
; ... your training code ...
(md "## Results" "Training curves and the best checkpoint.")
```

These are actual Hy macros: `lispyn run` treats them as documentation no-ops,
while `lispyn build` extracts them into notebook cells **without evaluating**
their content. Keep them at the top level and use literal strings.

## Full experiment: ResNet18, 10 + 300 epochs

A complete example using real TorchVision weights and your own labeled images:

```text
examples/data/resnet_images/
├── cats/
│   ├── cat_001.jpg
│   └── cat_002.jpg
└── dogs/
    ├── dog_001.jpg
    └── dog_002.jpg
```

The dataset can contain **any number of classes**; the names above are examples.
Provide enough images per class for a meaningful split. This is actual
fine-tuning of pretrained weights, **not** training a network from scratch.

```clojure
(require lispyn.macros [md])
(md "# ResNet18: image classification" "First tune the head, then the full model.")

(import torch)
(import torchvision)
(import lispyn.media [image-folder])
(import lispyn.data [split-dataset])
(import lispyn.reproducibility [seed-all])

(seed-all 42)
(setv weights torchvision.models.ResNet18_Weights.DEFAULT)
(setv images (image-folder "data/resnet_images" :transform (weights.transforms)))
(setv parts (split-dataset images :stratify True :seed 42))
(setv training (get parts 0))
(setv validation (get parts 1))
(setv model (torchvision.models.resnet18 :weights weights))
(setv model.fc (torch.nn.Linear model.fc.in_features (len images.classes)))

(md "## Stage 1: classifier fine-tuning"
    "Freeze the backbone and train only the classification head.")
(require lispyn.macros [head-tune])
(setv warmup
  (head-tune model training "fc"
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 10 :batch-size 16 :lr 0.001
    :metrics "accuracy" :experiment-dir "runs/resnet/head"))

(md "## Stage 2: deep/full fine-tuning"
    "Unfreeze the ResNet backbone and continue for **300 epochs**.")
(require lispyn.macros [full-tune])
(setv deep
  (full-tune warmup.model training
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 300 :batch-size 16 :lr 0.00001
    :metrics "accuracy" :experiment-dir "runs/resnet/deep"))

(md "## Training curves" "Compare training and validation performance.")
(require lispyn.macros [defcurve])
(defcurve resnet-chart
  :theme "paper" :train-color "#2563EB" :val-color "#D97706")
(resnet-chart deep :metric "loss" :title "ResNet18 loss"
  :save "runs/resnet/deep/loss.png")
(resnet-chart deep :metric "accuracy" :title "ResNet18 accuracy"
  :save "runs/resnet/deep/accuracy.png")
```

Full source: [`examples/resnet_finetune_300.hy`](examples/resnet_finetune_300.hy).
The notebook also contains authored Markdown explanations and saves separate
histories/checkpoints for both training phases.

**Important:** The 300-epoch notebook is generated, **not executed** in CI.
You need real class-labeled images and the appropriate PyTorch/TorchVision
installation. Pretrained weights may download on first execution, and a full
300-epoch run can be expensive. Avoid data leakage by grouping correlated
images before splitting; the built-in stratified split is sample-level only.

## Browse the generated notebooks

```bash
# Compile the curated examples; no downloads or model training
lispyn gallery

# Inspect the Python Jupyter notebooks
# example_generated_notebooks/*.ipynb

# Execute one small example and save its outputs in the gallery
lispyn execute examples/markdown_cells.hy \
  -o example_generated_notebooks/markdown_cells.ipynb
```

The repo includes a gallery under [`example_generated_notebooks/`](example_generated_notebooks/).
The collection includes an unexecuted ResNet18 notebook with the full
300-epoch training **code**, but never fabricated metrics or saved models.
## What works today

| Area | Current support |
| --- | --- |
| **Data** | CSV/JSON/Parquet, reproducible and stratified PyTorch dataset splits |
| **Images** | TorchVision image folders, transforms, pretrained-model recipes |
| **Text** | Text files, tokenization collators, Transformers and PEFT LoRA |
| **Audio** | TorchAudio decoding, labeled folders and spectrograms |
| **Training** | Head-only/full fine-tuning, LoRA, accuracy, checkpoints, metrics |
| **Reproducibility** | Seeds, recorded environment/configuration, run reports |
| **Jupyter** | Macro-authored Markdown, generated notebook gallery, reusable Matplotlib designs |

Examples: [`plot_experiment.hy`](examples/plot_experiment.hy), [`stratified_inference.hy`](examples/stratified_inference.hy), [`one_call_training.hy`](examples/one_call_training.hy), [`one_call_lora.hy`](examples/one_call_lora.hy), [`vision_finetune.hy`](examples/vision_finetune.hy), [`audio_features.hy`](examples/audio_features.hy). Optional integrations need their compatible libraries installed.

## Direction

1. **Keep source concise:** useful Lisp macros, no imitation of the PyTorch API.
2. **Keep experiments inspectable:** generated Python notebooks and explicit outputs.
3. **Keep the ecosystem open:** native Python functions, community macros, small tested releases.

**Experimental v0.11.0-alpha:** APIs may change. Recording a seed and package versions helps repeat an experiment but does not guarantee deterministic GPU execution. Notebook compilation may execute Hy macros: **run only trusted `.hy` files**. The run manifest does not store dataset contents or secrets; using the same data and base weights remains your responsibility.

For API details, optional installation and advanced examples, see the [extended guide](docs/guide.md). Contributions and issues are welcome.

## Author and license

Created by **Eduardo J. Barrios**. Copyright © 2026 Eduardo J. Barrios. Licensed under **Apache-2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Redistribution must preserve applicable notices. The license does not require attribution merely for executing a program. Generated notebooks include author attribution by default.
