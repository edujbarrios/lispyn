# Lispyn — Extended Guide

**Lisp macros for Python, data science, PyTorch and Jupyter.**

Lispyn is an experimental, open-source project originally created by **Eduardo J. Barrios**. Its purpose is to make Jupyter experimentation more concise and expressive through Lisp macros and reusable Python tooling.

Lispyn extends [Hy](https://hylang.org/) (a Lisp running on Python). It does **not** replace Python, hide PyTorch, or require developers to reimplement familiar ML APIs. Import Python modules and call their functions directly.

> Status: **v0.25.0a1 (experimental)**. No stability guarantees; published as an experimental alpha on [PyPI](https://pypi.org/project/lispyn/).

## Install from source

Requires Python 3.10+:

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
python -m pip install -e '.[notebook,data]'
```

Install PyTorch separately with the [official CPU/GPU instructions](https://pytorch.org/get-started/locally/). For Parquet loading, install `pyarrow` as well.

## Tell the story of an experiment with Hy macros (0.24.0a2)

Lispyn is designed to make complex experiments **understandable, reusable and
shareable**, not just executable. Declare editable notebook parameters and
generate a readable results report without dropping out of Hy:

```hy
(require lispyn.macros [md])
(md "# A reproducible experiment"
    "State your hypothesis, dataset and evaluation method here.")

(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 10)
  (setv learning-rate 0.001))

; In the next notebook code cell, train a native PyTorch model and obtain run.
(require lispyn.macros [report-run])
(report-run run
  :title "Baseline experiment"
  :notes "What changed, what worked, and what remains uncertain."
  :save "runs/baseline.md")
```

**`experiment-params`** compiles ordinary Hy `(setv name value)` forms into
a standard Jupyter code cell tagged `parameters`. Compatible parameter-cell
tools can override these defaults without editing the notebook source. The
same source runs directly with `lispyn run`, where assignments execute
normally. Put the form on its own lines at the top level; include one or more
`setv` assignments. For reusable/overridable defaults, prefer simple
Python-compatible literals (strings, numbers, booleans).

**`report-run`** accepts a Lispyn training Run or a mapping of metric
histories. It renders Markdown in Jupyter, prints Markdown when run from a
terminal, and optionally exports a `.md` artifact. The report identifies
best/final metric values, optimal epoch, train/validation metrics and early
stopping where available. It includes **no training data, model weights or
predictions**; if you add notes, review them before sharing. The default
best-value direction is to minimize loss/error metrics and maximize other
metrics; override with `:metric "val_loss" :mode "min"` when needed.

Start a new narrative-style experiment with
`lispyn new my_experiment.hy`. See the
[illustrative Hy source](../examples/experiment_story.hy) and regenerate a
Jupyter notebook using `lispyn gallery` or
`lispyn build examples/experiment_story.hy`. The example's metrics are
**synthetic illustration**, not the outputs of model training.

Each generated code cell compiles independently. Reuse `(require ...)` in
a later Hy cell if it calls a macro; the parameter cell itself includes its
own compiled macro import.

## Reuse a Hy experiment with parameter overrides (0.25.0a1)

Lispyn now makes the `experiment-params` macro useful beyond editing notebook
cells: **inspect parameters, build a configured notebook, execute a configured
notebook, or generate a small grid of variants from one unchanged `.hy` file.**
The authoring language stays Hy/Lisp; Python/Jupyter are output formats.

```hy
(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 5)
  (setv learning-rate 0.001))

; Write the rest of your experiment in Hy, using those variables.
```

```bash
# Inspect declared defaults, without running any model:
lispyn parameters examples/parameter_variants.hy --json

# Generate a notebook using different experimental parameters:
lispyn build examples/parameter_variants.hy \
  --set epochs=12 --set learning-rate=0.01 -o runs/variant.ipynb

# Or read a JSON object of overrides from disk and execute the notebook:
lispyn execute examples/parameter_variants.hy \
  --params overrides.json --set epochs=8 -o runs/executed.ipynb

# Build (but do not execute) a reproducible parameter grid:
lispyn sweep examples/parameter_variants.hy \
  --grid grid.json -o runs/variants
```

Example `overrides.json`: `{"seed": 17, "epochs": 6}`. Example `grid.json`:
`{"epochs": [3, 6], "learning-rate": [0.001, 0.01]}`. The grid produces four
standard, **unexecuted** Jupyter notebooks and a `sweep.json` index of
their chosen parameter values. At most 64 variants are allowed per sweep,
and existing variants/indexes are never overwritten.

Overrides are **strict JSON literals**, not executable expressions.
Strings require JSON double quotes (e.g. `--set 'label="baseline"'`);
objects and arrays are also accepted. Parameter names may use either
Hy hyphens or compiled Python underscores (e.g. `learning-rate` and
`learning_rate` refer to the same parameter). An unknown, duplicated,
non-finite or malformed override is rejected. `--set` takes precedence
over matching values in `--params`.

Every notebook includes a SHA-256 hash of the original Hy source in
`metadata.lispyn.source_sha256`. The source file is never modified;
variant notebooks have deterministic cell IDs and record the **names**
of overridden parameters in metadata. The sweep index contains the override
**values** for reproducibility: review both notebooks and index before
sharing, especially if you put sensitive information in parameters.

Compilation can execute trusted Hy *compile-time macros*. Neither
`lispyn parameters` nor `lispyn sweep` executes the notebook or trains
a model. Only `lispyn execute` runs the generated notebook. `lispyn run`
remains a direct Hy execution path without parameter-override flags.

See [the full Hy experiment](../examples/parameter_variants.hy) and
`lispyn gallery` for a compiled notebook. It uses *illustrative*
metrics, not real model training.

## Run your first experiment

```bash
# Generates a .ipynb without executing cells
lispyn build examples/dataset_experiment.hy

# Generates and executes the .ipynb, recording results
lispyn execute examples/dataset_experiment.hy

# Executes the original Hy program; no notebook is created
lispyn run examples/dataset_experiment.hy

# Execute package tests
python -m pip install -e '.[dev,data]'
pytest -q
```

The tiny regression example reads `examples/data/tiny_regression.csv`, removes incomplete rows, splits train/validation, creates native PyTorch `TensorDataset` and `DataLoader` objects, trains a `torch.nn.Linear`, and prints validation loss. It does not download data or require a GPU. A simpler single-cell example remains in [`examples/first_experiment.hy`](examples/first_experiment.hy).

## Repeated cross-validation from Lisp macros (0.22.0a1)

Lispyn's new **`cross-validate`** macro runs a Hy callback on each
training/validation fold and returns a compact JSON-ready report. The callback
creates a fresh model and optimizer per fold; PyTorch remains a native backend.
Nothing from the held-out test set is passed to cross-validation.

```hy
(require lispyn.macros [cross-validate train-until save-cross-validation])
(import torch)

; `training` and `train-labels` come from a previous train-val-test split.
(defn score-fold [fold-training fold-validation fold repeat]
  (setv model (torch.nn.Linear 2 2)) ; new model for every fold
  (setv run (train-until model fold-training 2
    :validation fold-validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :metrics "accuracy"
    :epochs 4 :batch-size 8
    :seed (+ 42 (* repeat 100) fold)))
  (get (get run.history "val_accuracy") -1))

(setv summary (cross-validate training score-fold
  :n-splits 4 :n-repeats 2 :labels train-labels
  :metric "val_accuracy" :seed 42))
(print "Fold mean:" (get summary "mean"))
(print "Weighted mean:" (get summary "weighted_mean"))
(save-cross-validation summary "runs/cv_report.json")
```

The `cross-validate` and `cross-validate-table` macros support one-off and
repeated k-fold, optional stratification and **group-aware** folds. Repeated
splits use different deterministic seeds. Each sample is held out exactly once
*per repeat*. `weighted_mean` accounts for different validation fold sizes,
which can arise with indivisible subjects/sessions. Reports include repeat,
fold, split sizes, metric and aggregate scores, **not dataset rows or model
predictions**. A score callback must return one finite real number; it should
create a new model and optimizer each time.

For lower-level control use `repeated-cross-validation-folds` (PyTorch) or
`repeated-tabular-folds` (pandas), yielding
`[repeat fold training validation]`. Tabular convenience macros
`tabular-train-val-test` and `tabular-cross-validation-folds` are also
available. Group labels and class labels are mutually exclusive, and must
align with the **input dataset**; when working on a PyTorch `Subset`, select
the appropriate `labels` or `groups` by its `indices`.

See the [full Hy example](../examples/repeated_cross_validation.hy), compiled
by `lispyn gallery`, or run it directly using
`lispyn run examples/repeated_cross_validation.hy` (requires PyTorch).

## Chronological splits and walk-forward CV in Lisp (0.23.0a1)

For time-dependent datasets, random k-fold can leak future observations into
training. Lispyn now offers **Hy-first time-series macros** that preserve
chronological order, optionally leave a sample-level **gap** between sets, and
support both expanding and fixed-size rolling windows.

```hy
(require lispyn.macros
  [temporal-train-val-test temporal-folds cross-validate-temporal])

; `dataset` is an indexable PyTorch dataset ordered oldest to newest.
(setv partitions (temporal-train-val-test dataset
  :validation-fraction 0.2 :test-fraction 0.1 :gap 2))
(setv history (get partitions 0))
(setv final-test (get partitions 2)) ; untouched during model selection

; Each fold trains only on observations earlier than its validation window.
(for [[fold-train fold-valid]
      (temporal-folds history
        :n-splits 3 :test-size 4 :gap 1 :max-train-size 12)]
  (print "Train / validation:" (len fold-train) (len fold-valid)))

; The callback makes a fresh model per fold and returns one finite score.
(setv report (cross-validate-temporal history score-window
  :n-splits 3 :test-size 4 :gap 1 :metric "val_loss"))
(print (get report "mean"))
```

The `temporal-train-val-test`, `temporal-folds`, and
`cross-validate-temporal` macros work with native PyTorch datasets. Pandas
users have `temporal-table-split`, `temporal-table-folds` and
`cross-validate-temporal-table`; pass `:time-column "date"` to sort
ascending by a **unique, nonmissing** timestamp. Without `:time-column`,
the existing row order is treated as chronological.

An optional `:gap N` excludes N samples immediately before every validation
window (and before the final test partition in a three-way split).
The gap counts **rows, not elapsed time**, and cannot protect against features
already constructed using future information. Three-way fractions apply after
removing two gaps. For CV, `:max-train-size N` enables rolling rather than
expanding windows; the earliest training prefix is not validated. Time-series
CV is **not** standard k-fold: each validation block is evaluated exactly
once in temporal order, and training can reuse past observations.

The Hy callback must initialize a fresh model and optimizer for each fold.
See the [complete Hy/PyTorch recipe](../examples/temporal_cross_validation.hy),
also compiled to a Jupyter notebook by `lispyn gallery`. Previous random,
stratified, grouped and repeated CV macros are unchanged.

## Data utilities

Import functions from `lispyn.data` inside Hy (Hy translates names like `read-dataset` to Python's `read_dataset`):

```clojure
(import torch)
(import lispyn.data [read-dataset drop-missing train-validation-split
                     dataframe-to-dataset make-dataloader])

(setv df (drop-missing (read-dataset "data/train.csv")))
(setv parts (train-validation-split df :validation-fraction 0.2 :seed 42))
(setv training (get parts 0))
(setv validation (get parts 1))
(setv train-dataset (dataframe-to-dataset training :features ["x1" "x2"] :target "label" :target-dtype torch.long))
(setv train-loader (make-dataloader train-dataset :batch-size 32 :shuffle True :seed 42))
```

- `read_dataset`: CSV, JSON, JSONL, Parquet via pandas (optional extra `data`; Parquet may need pyarrow).
- `drop_missing`: return a new DataFrame without rows missing selected values.
- `train_validation_split`: seeded row-level split (not stratified, grouped, or time-aware).
- `dataframe_to_dataset`: numeric features + numeric target into a native PyTorch `TensorDataset`; accepts PyTorch dtypes.
- `make_dataloader`: standard PyTorch `DataLoader` with optional seeded shuffle; forwards other DataLoader keyword arguments.
- `to_tensor`: `torch.as_tensor` convenience function.

**Prevent data leakage**: choose the correct split for groups or time series and fit data-dependent transforms only on training data. Lispyn's default split is designed for small independent-row experiments, not every dataset.

You can always use *any* Python or PyTorch function normally through Hy; Lispyn does not whitelist or wrap their APIs.

## Multiple notebook cells

Use source comment markers to request Jupyter Markdown and Python code cells. Markers are comments, so the `.hy` file remains executable by Hy:

```clojure
; %% [markdown]
; ## Setup
; These lines become notebook Markdown.
; %% [code]
(import torch)
(setv x (torch.tensor [1 2 3]))
; %% [markdown]
; ## Results
; %% [code]
(print x)
```

`lispyn build FILE.hy` creates a `.ipynb` with project attribution and the described cells. `lispyn execute FILE.hy` executes all code cells in the same Python kernel and saves the output notebook. Relative paths resolve from the `.hy` file's directory.

**Important:** Each marked code section is compiled independently by `hy2py` but shares *runtime Python state* when executed in the notebook. If a section uses Hy macros, include `(require ...)` within that section, even if an earlier section did so. Ordinary variables imported or assigned in an earlier code section remain accessible in later cells. Without markers, Lispyn still compiles the whole source into **one** code cell.

## Security, execution and reproducibility

- `build` does **not** run the generated Python notebook, but **Hy macros can execute arbitrary code during compilation**. Do not build untrusted `.hy` sources.
- `execute` runs notebook code through `nbclient` and writes `.ipynb` results. `run` executes Hy directly.
- `with-seed` seeds Python's `random` and PyTorch, but it cannot guarantee strict determinism on every GPU, PyTorch operation, or backend.
- Notebook generation does not launch the Jupyter web server.

## Roadmap

- [x] Installable Python package and CLI (`build`, `execute`, `run`)
- [x] Hy-to-Python Jupyter notebook generation and execution
- [x] Dataset readers and reproducible train/validation splitting
- [x] PyTorch TensorDataset / DataLoader adapters
- [x] Multi-cell notebooks with Markdown and code sections
- [x] Training, evaluation, checkpoint helpers and a train-for macro
- [x] ImageFolder, text datasets with tokenization, and labeled audio folders
- [ ] More dataset formats and stratified/grouped splitting
- [x] Optional Hugging Face PEFT/LoRA adapter utilities and offline text demo
- [x] Pretrained TorchVision recipe and genuine WAV decoding tests
- [x] Metrics, JSON history and automatic best-checkpoint saving
- [x] One-call PyTorch and LoRA training with automatic optimizer/device
- [x] Typed labels for CSV text datasets
- [ ] Reproducibility reports and richer error diagnostics


## Images, text, audio and fine-tuning (v0.3 alpha)

Lispyn keeps the full native PyTorch, TorchVision and TorchAudio APIs available through Hy.

- `image_folder("images/", transform=...)`: wraps `torchvision.datasets.ImageFolder` for `images/class_name/*.jpg`. For pretrained models, supply the weights' recommended transforms.
- `text_dataset("train.jsonl", text_key="text", label_key="label")`: reads `.txt`, `.csv`, `.json`, `.jsonl`. Use `token_collate(tokenizer)` with compatible tokenizers (e.g. Hugging Face) inside a `DataLoader`.
- `audio_folder("audio/", sample_rate=16000, num_frames=16000)`: labeled folder dataset, backed by `torchaudio.load`; supports resampling and fixed-length waveforms, with directory layout `audio/class_name/clip.wav`.
- `trainable(model, "fc")`: freeze other parameters and train only the named layer. `trainable(model)` unfreezes all parameters for full fine-tuning. Create your optimizer **after** changing which layers are trainable.
- `fit(model, loader, optimizer, loss_fn=..., epochs=..., validation=...)`, `evaluate(...)`, `save_checkpoint(...)`, `load_checkpoint(...)`: train, validate and save PyTorch state dicts. For nonstandard model/batch formats use a custom `step(model, batch)` callback.

Run a minimal model using TorchVision's synthetic `FakeData`—no data downloads:

```bash
lispyn execute examples/vision_finetune.hy
```

Choose the correct matching versions of `torch`, `torchvision`, `torchaudio` and optional `torchcodec` for your environment, especially for GPU installs. In recent TorchAudio versions, media decoding depends on TorchCodec and often FFmpeg; see the [official installation guide](https://docs.pytorch.org/audio/stable/installation).

These v0.3 helpers cover classifier/head-only and full fine-tuning. v0.4 adds PEFT/LoRA separately; QLoRA and distributed training are still out of scope.

## LoRA adapters, pretrained vision and real audio (v0.4 alpha)

**Use native PyTorch. Add only the abstraction you need.**

Install optional text/adapter dependencies (choose the correct torch build for your CPU/GPU first):

```bash
python -m pip install -e ".[peft]"
```

A short Hy example for **LoRA fine-tuning**:

```clojure
(import torch)
(import lispyn.peft_utils [lora parameter-counts save-adapter])
(import lispyn.training [fit])

(setv model (lora model ["query" "value"] :rank 8 :task-type "SEQ_CLS"))
(setv optim (torch.optim.AdamW
  (filter (fn [p] p.requires_grad) (.parameters model)) :lr 0.0001))
(setv history (fit model batches optim :epochs 3))
(save-adapter model "adapters/experiment")
```

Here `model` is an existing compatible Transformers model and `batches` is a loader returning mappings with `input_ids`, `attention_mask`, and `labels`. LoRA targets depend on the model architecture. A small, completely offline BERT demonstration lives at [examples/text_lora.hy](examples/text_lora.hy).

- `lora(model, ["query", "value"], rank=8, task_type="SEQ_CLS")` uses **Hugging Face PEFT**; it is not a new LoRA implementation. The original model is modified by PEFT.
- `parameter_counts(model)` reports the trainable/total parameters and percentage.
- `save_adapter(model, folder)` and `load_adapter(base_model, folder)` save/load LoRA adapters, not complete pretrained weights. You must retain or re-download the compatible base model separately.
- Full fine-tuning is still available: `trainable(model)` unfreezes all parameters. For head-only training, `trainable(model, "fc")` freezes the rest.

### Examples

```bash
# Tiny BERT + LoRA, no model download (requires optional peft extra)
lispyn execute examples/text_lora.hy

# Native TorchAudio mel spectrogram, no downloads
lispyn execute examples/audio_features.hy

# Real ResNet18 transfer learning: supply photos/class_name/*.jpg
# (downloads the official pretrained weights on first use)
lispyn execute examples/vision_transfer.hy
```

For image inference and transfer learning use `torchvision.models.*_Weights.DEFAULT` with `weights.transforms()` to match pretrained normalization and resize. See [TorchVision's model guide](https://docs.pytorch.org/vision/main/models.html).

For audio reading, current TorchAudio `load()` delegates to **TorchCodec**. Install matching `torch`, `torchaudio`, `torchcodec` and system FFmpeg support as needed. The automated test suite generates an actual WAV file and checks decode, resampling and AudioFolder loading, in addition to the synthetic spectrogram example. See [TorchAudio installation notes](https://docs.pytorch.org/audio/stable/installation).

**Scope:** LoRA fine-tuning, classifier-only tuning, and full tuning work on supported models. QLoRA/quantization, distributed training, speech recognition pipelines, streaming datasets and automatic task inference are not provided yet. The user's data/models may contain sensitive content and must be handled according to their permissions and licenses.

## Training with metrics and checkpoints (v0.5 alpha)

Train image, text or audio models through the same small helper, while using
native PyTorch datasets, modules, optimizers and loss functions.

```clojure
(import torch)
(import lispyn.training [fit accuracy])

(setv history
  (fit model train-loader optimizer
    :loss-fn (torch.nn.CrossEntropyLoss)
    :validation val-loader
    :epochs 5
    :metrics "accuracy"
    :save-best "checkpoints/best.pt"
    :log-path "history.json"))
```

Here `model`, `train-loader`, `val-loader` and `optimizer` are your existing
PyTorch objects. The complete runnable demonstration is in
[examples/compact_training.hy](examples/compact_training.hy). Run it with:

```bash
lispyn execute examples/compact_training.hy
```

**New `fit()` options:**

- `metrics="accuracy"` or `metrics={"metric_name": metric_fn}`; a custom
  metric receives `(logits, labels)` and returns one numeric batch score.
- `save_best="best.pt"` writes model and optimizer state on improvement;
  the default monitor is `val_loss` when validation is present, otherwise
  `train_loss`. Set `monitor="val_accuracy", mode="max"` to prefer accuracy.
- `log_path="history.json"` writes metric and loss arrays after every epoch.
- `evaluate_metrics(...)` returns loss plus optional metrics; existing
  `evaluate(...)` still returns a single float. `load_checkpoint(...)`
  restores the checkpoint.

The checkpoint represents the **best epoch**. Training continues afterward,
so the in-memory model ends at the final epoch; restore the best one explicitly.
History values are **unweighted means of per-batch** values, not dataset-wide
sample-weighted statistics. The built-in accuracy metric expects multiclass
logits `[batch, classes]` and class labels `[batch]`. A custom `step`
returns a scalar loss and cannot supply prediction metrics in this version.

For PEFT models, a regular `save_best` checkpoint stores the complete
`state_dict`. Use `save_adapter(...)` separately when you want a compact
adapter-only artifact. The base model must be available when restoring adapters.

## One-call experiments (v0.6 alpha)

Lispyn now provides `train()` in `lispyn.workflow`: it accepts native PyTorch
`Dataset`, `DataLoader`, or an iterable of already prepared batches, creates
a native optimizer, optionally applies head-only or PEFT LoRA fine-tuning,
then delegates to `fit()`. **You can always call PyTorch directly**.

```clojure
(import torch)
(import lispyn.workflow [train])

(setv result
  (train model dataset
    :loss-fn (torch.nn.CrossEntropyLoss)
    :validation valid-dataset
    :epochs 5
    :lr 0.0001
    :metrics "accuracy"
    :save-best "best.pt"))
(print result.history)
```

The `model`, `dataset` and `valid-dataset` here are ordinary PyTorch objects.
For a full working offline example, use:

```bash
lispyn execute examples/one_call_training.hy
```

Choose `head="fc"` to train the named classifier head, or use
`lora_modules=["query", "value"], lora_rank=8, lora_task_type="SEQ_CLS"`
for LoRA through optional Hugging Face PEFT. These options cannot be combined.
Use the offline `examples/one_call_lora.hy` to exercise the LoRA path.

`train()` returns an object containing `.model`, `.optimizer`, `.history`,
and `.device`. The model may be a new PEFT-wrapped model, so use the returned
`.model` after LoRA. The default optimizer is AdamW; alternatives are
`"adam"`, `"sgd"` or a PyTorch optimizer class. Arbitrary optimizer arguments
can be forwarded in `optimizer_kwargs`.

`device="auto"` selects CUDA, then MPS, then CPU according to PyTorch's
availability checks. An explicit `device="cpu"` is helpful when debugging
or producing CPU-only notebook examples.

**Text CSV labels:** `text_dataset("train.csv", label_key="label", label_cast=int)`
converts numeric CSV labels to integers for `token_collate(...)`. A custom
`label_cast` callable can map named labels. Without it, CSV labels remain
strings, as read by Python's CSV parser.

**Scope:** This wrapper does not choose a loss function automatically, download
models, tokenize text automatically, quantize models, or hide PyTorch features.
The dataset split must be prepared separately and should not leak samples
between training and validation. For standard PEFT adapter-only saves use
`save_adapter()` separately from `save_best` full state dictionaries.

## Contributing

Issues and pull requests are welcome. Keep changes small, documented, and tested. Install development dependencies with `python -m pip install -e '.[dev,data]'` and run `pytest`.

## License and attribution

Copyright © 2026 **Eduardo J. Barrios**. Licensed under the **Apache License 2.0**; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Redistributions must preserve applicable copyright and attribution notices under that license. Apache-2.0 does **not** require citation merely for running Lispyn or using a generated notebook. Notebooks include author attribution by default.



For the new experiment manifest and report command, see the project [README](../README.md).


## Reproducible stratified dataset splits (v0.8 alpha)

Use native `torch.utils.data.Subset` objects. No dataset copying, no sklearn dependency.

```clojure
(import lispyn.data [split-dataset])
(import lispyn.media [image-folder])

(setv photos (image-folder "photos"))
(setv halves (split-dataset photos :stratify True :seed 42))
(setv training (get halves 0))
(setv validation (get halves 1))
```

`ImageFolder`, labeled `audio_folder` and labeled `text_dataset` expose
`.targets` for convenience. For a `TensorDataset`, pass
`:labels labels` explicitly: `(split-dataset data :labels labels)`.
Split is deterministic for the same label order and seed, and every class
must contain at least two samples. The per-class rounding may change the
overall validation percentage. **It is not group-aware**: avoid leakage between
train and validation when samples share a person, recording, document,
subject, session, or other correlated source.

## Inference as a Lisp macro

```clojure
(require lispyn.macros [with-eval])
(with-eval model
  (setv outputs (model batch))
  (print outputs.shape))
```

`with-eval` temporarily calls `model.eval()`, runs the body under
`torch.inference_mode()`, and restores the previous `model.training`
state even if the body raises an exception. This is a **real compile-time
Hy macro**, not a Python wrapper. Use only for inference, not training.

The runnable offline example is
[`examples/stratified_inference.hy`](../examples/stratified_inference.hy).

## Visualize experiments in Jupyter (v0.9 alpha)

Install optional plotting support with `python -m pip install -e ".[plots]"`.
No Matplotlib dependency is needed for `lispyn build` or ordinary model training.

```clojure
(import lispyn.plots [plot-history])
(plot-history result :metric "loss" :save "artifacts/loss.png")
(plot-history result :metric "accuracy" :save "artifacts/accuracy.png")
```

The Python helper accepts the result of `lispyn.workflow.train`,
a metrics-history dictionary, or a path to `history.json`.
It plots **recorded** values only; it does not repeat training or evaluation.
If the metric is absent or includes non-finite values, it fails with a
clear error. One figure shows train/validation curves for a single metric,
so metrics on different scales are not mixed on one axis.

The CLI offers the same view without opening Jupyter:

```bash
lispyn report examples/runs/smoke --plot examples/runs/smoke/loss.png
lispyn report examples/runs/smoke --metric accuracy --plot examples/runs/smoke/accuracy.svg
```

See [`examples/plot_experiment.hy`](../examples/plot_experiment.hy) for an offline example.


## Reusable Matplotlib curve macros (v0.10 alpha)

Reusable designs are the main reason to add a plotting macro instead of
repeating PyPlot setup and labels in every notebook.

```clojure
(require lispyn.macros [curve defcurve])

; Define a named design once, then reuse it
(defcurve report-curve
  :theme "paper"
  :train-color "#2563EB"
  :val-color "#D97706"
  :xlabel "Epoch"
  :train-label "Training"
  :val-label "Validation")

(report-curve result :metric "loss" :title "Loss curve"
  :ylabel "Cross entropy" :save "figures/loss.png")
(report-curve result :metric "accuracy" :title "Accuracy curve"
  :save "figures/accuracy.svg")

; For a one-off curve, without creating a named design
(curve result :metric "loss" :theme "dark" :title "Dark report"
  :train-color "cyan" :val-color "#FBBF24")
```

`curve` and `defcurve` are **Hy compile-time macros**, not aliases for Python
functions. `defcurve` emits a reusable plotter; options passed when using
it override defaults. Both delegate rendering to `lispyn.plots.plot_history`
and use the original Matplotlib Figure object.

Three built-in themes are `clean` (default), `dark`, and `paper`. Supported
options include `:title`, `:xlabel`, `:ylabel`, `:train-label`, `:val-label`,
`:train-color`, `:val-color`, `:figsize [8 4]`, `:linewidth 2`,
`:grid True`, `:legend True`, `:dpi 140`, `:save "chart.png"`.

Themings are local to each figure; they do not modify global Matplotlib
configuration. As each Hy notebook code cell compiles independently,
put `(require lispyn.macros [curve defcurve])` in the cell where you use
the macros. A plotter defined in one executed cell remains available as
a Python object to later cells.

CLI reports can also control theme, text and colors:

```bash
lispyn report runs/demo --plot runs/demo/curves.svg --metric loss \
  --theme dark --title "Experiment loss" \
  --train-color '#22D3EE' --val-color '#FBBF24'
```

See [examples/curve_designs.hy](../examples/curve_designs.hy).

## Real notebook Markdown macros and generated example gallery (v0.11 alpha)

The Markdown macros accept literal strings as top-level forms:

```clojure
(require lispyn.macros [md markdown])
(md "# Notebook title" "Brief explanation with **formatting**.")
(print 42)
(markdown "## Evaluation" "Scores and charts.")
```

`lispyn build` translates them into normal Markdown cells; it never evaluates
the Markdown. `lispyn run` directly runs the Hy source: the macros are
documentation no-ops. Dynamic expressions, nested forms and directives sharing
a source line with other Hy forms are intentionally rejected. If using the
older `; %% [markdown]` / `; %% [code]` markers, they remain supported.

**Notebook compilation happens code cell by code cell.** Repeat the required
`(require lispyn.macros [...])` in each cell where its macro is used.
Runtime variables survive between cells, just as in a normal Jupyter session.

Gallery: `lispyn gallery` compiles curated sources from `examples/` into
`example_generated_notebooks/`. The command does **not** run the notebook.
Each notebook has stable cell IDs and includes a Markdown author notice.
Run a selected experiment with `lispyn execute examples/name.hy -o
example_generated_notebooks/name.ipynb` to save *real* execution outputs.

### Two-stage ResNet18 fine-tuning

See [examples/resnet_finetune_300.hy](../examples/resnet_finetune_300.hy)
for a full model, data, Markdown, metrics and plotting pipeline.
It expects `examples/data/resnet_images/<class_name>/*.jpg` (or PNG) and
pretrained weights downloaded by TorchVision on first execution.

The helper macros `(head-tune model train-data "fc" :epochs 10 ...)`
and `(full-tune model train-data :epochs 300 ...)` delegate to native
PyTorch via `lispyn.workflow.train()`. The first freezes backbone parameters;
the second unfreezes every parameter and starts a fresh optimizer at a
lower learning rate. They support any compatible native PyTorch model
(not just ResNet). Separate experiment directories retain individual
checkpoints and training metrics. The full 300-epoch run is **not** performed
by the test suite, and compiled examples do not contain artificial results.

## Install or use Python libraries from Hy (v0.12 alpha)

Python standard-library modules such as `math` already work with `(import math)`.
For named third-party dependencies, Lispyn offers runtime-only macros:

```clojure
(require lispyn.macros [ensure-package use-package])
(ensure-package "matplotlib>=3.8")
(ensure-package "scikit-learn" :import-name "sklearn")
(setv math (use-package "math"))
(print (math.sqrt 25))
(setv np (use-package "numpy"))
(print (np.arange 4))
```

`ensure-package` checks the *currently running* Python/Jupyter interpreter and
prints `Requirement already installed in this environment: <package>` when
available. It installs absent requirements via `sys.executable -m pip` with
explicit arguments, **never a shell**. Use `:install False` to fail instead
of installing. `use-package` performs the check and returns the imported
Python module; distribution and module names may differ, hence `:import-name`.

These macros are compiled into standard Python notebook cells. They do not
install anything when calling `lispyn build` or `lispyn gallery`; installation
only happens during **execution** (`lispyn execute` or running notebook cells).
A Jupyter kernel restart may be necessary after installing compiled packages.
Only run trusted notebooks with dependencies you intend to install.

Requirements accept distribution names and optional version specifiers, but
not direct URLs or environment markers. Already-installed distributions are
checked against version constraints. For GPU-specific Torch packages, install
a compatible build following official instructions beforehand. These helpers
do not perform GPU wheel selection or dependency environment isolation.

Use `(import math)` or `(import your_library)` directly if packages are
already available; these helpers are optional conveniences, not a replacement
for Hy's normal Python interoperability.

## Dedicated pip-install cells and a synthetic plotting demo (v0.13 alpha)

Write `(pip-install "matplotlib>=3.8" "numpy")` at the **top level**
of a Hy file, on its own lines, after `(require lispyn.macros [pip-install])`.
`lispyn build` turns this form into a **dedicated Python Jupyter code cell**,
tagged `lispyn-pip-install`, containing a call to `ensure_packages`.
Installed requirements and standard-library modules are reported and skipped.
Missing named packages are installed only on **notebook execution** using
the current kernel's Python executable. Do not install arbitrary dependency
lists from untrusted notebooks; packages may run installation scripts.

```clojure
(require lispyn.macros [pip-install])
(pip-install "math" "packaging>=23" "matplotlib>=3.8")

; Code after the setup cell:
(require lispyn.macros [use-package])
(setv math (use-package "math" :install False))
(print (math.sqrt 16))
```

`pip-install` accepts only literal named package strings; options such as
pip flags, URLs or dynamic expressions are rejected in compiled notebook
cells. If a package is missing, you can opt out of automatic installation
using the `ensure-package` macro with `:install False` instead.

Hy-to-Python compilation and gallery generation do not trigger pip.
Any newly installed native extension may require restarting the Jupyter
kernel before import. For CUDA builds use the official PyTorch installation
command in a suitable virtual environment beforehand.

The [synthetic plotting example](../examples/synthetic_curves.hy) includes
the pip-install macro, a named Matplotlib theme and a deterministic synthetic
loss history. Run `lispyn run examples/synthetic_curves.hy` to regenerate
`docs/images/synthetic_training_curves.png` for the README.

## Dark plots, turquoise palettes and multi-run comparisons (v0.15 alpha)

`plot_history`, `curve` and `defcurve` now support the theme
`docusaurus-dark` (dark gray background, subtle grid, light text) and the
seven-color `turquoise-spectrum` palette. Named designs remain reusable:

```clojure
(require lispyn.macros [defcurve compare-curves])
(defcurve dark :theme "docusaurus-dark"
  :palette "turquoise-spectrum" :title-size 18
  :line-width 2.8 :fill-alpha 0.08)
(dark run :metric "loss" :subtitle "Validation vs training"
  :save "loss.png")

; Compare multiple experiments with one macro:
(compare-curves {"baseline" baseline-history "finetuned" tuned-history}
  :metric "loss" :split "val" :theme "docusaurus-dark"
  :palette "turquoise-spectrum" :save "comparison.svg")
```

The plotting API additionally accepts `:figure-bg`, `:axes-bg`,
`:grid-color`, `:text-color`, `:spine-color`, `:legend-bg`,
`:legend-edge`, `:train-color`, `:val-color`, `:subtitle`,
`:title-size`, `:label-size`, `:tick-size`, `:marker`, `:alpha`,
`:fill-alpha`, `:line-width`, `:figsize`, `:dpi` and label options.
All options are scoped to the generated figure; Lispyn does not change
global `matplotlib.rcParams`. The comparison helper plots a requested
`train_` or `val_` metric from any number of named runs, with independent
epoch counts. Only recorded values are plotted.

The [synthetic plotting example](../examples/synthetic_curves.hy)
generates both PNGs embedded in the README. Both are deterministic
**synthetic examples** and require no downloads or model training.

## Classification evaluation and reusable confusion matrices (v0.16 alpha)

Class labels and predictions can be plotted without scikit-learn. Use the
`confusion-matrix` and `defconfusion` **compile-time Hy macros**:

```clojure
(require lispyn.macros [confusion-matrix defconfusion])
(defconfusion my-report :theme "docusaurus-dark"
  :palette "turquoise-spectrum" :title-size 16)
(my-report actual predicted :classes ["cat" "dog"]
  :normalize "true" :title "Classifier results" :save "matrix.png")

; Or create one plot directly:
(confusion-matrix actual predicted :normalize "all" :save "all.png")
```

`lispyn.classification.confusion_counts(actual, predicted, classes=None)`
returns the raw count matrix and ordered class labels. The source accepts
one-dimensional arrays, lists or scalar tensors. Class order defaults to
the first observed occurrence, unless `:classes` explicitly sets the order.
Unknown observed classes are rejected rather than silently excluded.

`normalize` supports `None` (counts), `"true"` (per actual class),
`"pred"` (per predicted class) and `"all"` (whole dataset). Zero-count
denominators produce zeros. The optional plotting dependency is Matplotlib
only, not scikit-learn. The result is a native Matplotlib Figure.

See [`examples/confusion_matrix.hy`](../examples/confusion_matrix.hy) for a
synthetic example and its generated image in `docs/images/`.

## Early stopping and last-epoch checkpoints (v0.17 alpha)

Use `train-until` to express a reusable validation-guided training policy
from Hy without replacing PyTorch's training loop:

```clojure
(require lispyn.macros [train-until])
(setv result
  (train-until model dataset 6
    :validation validation-dataset
    :loss-fn criterion :epochs 100 :min-delta 0.001
    :monitor "val_loss" :device "auto"
    :experiment-dir "runs/short"))
(print result.stopped-early)
```

`patience` is the count of **consecutive epochs without a significant
improvement** on the monitored metric. `min_delta` is the minimum absolute
improvement to consider meaningful (default 0). The monitor defaults to
`val_loss` with validation data, otherwise `train_loss`. As before, it can
be set to a tracked custom metric such as `val_accuracy` and `mode` can
be `min` or `max`. If patience is omitted, all requested epochs run.

When enabled with `experiment_dir`, a separate `last.pt` is written after
**each completed epoch** alongside `best.pt`, `history.json` and
`manifest.json`. The last checkpoint contains model weights, optimizer
state and history. The best checkpoint tracks the selected metric, using
the same `min_delta` threshold. Both checkpoints are native PyTorch
state dictionaries; loading one requires trusted files.

These are **not** automatic resumption or full GPU determinism features.
Early stopping does not automatically restore best weights in the live
model; use `load_checkpoint` to do that explicitly. Saving the last
checkpoint every epoch consumes extra disk space and may slow large
experiments, so it is only automatic when both `experiment_dir` and
`patience` are supplied. You can supply `save_last` explicitly as well.

The manifest records `epochs_completed` and `stopped_early`, and
`lispyn report runs/short` shows them. The [offline notebook example]
(../examples/early_stopping.hy) uses a tiny PyTorch dataset.

## v0.18: classification metrics, notebook metadata and run comparisons

### Class precision, recall, F1 and report exports

```clojure
(require lispyn.macros [classification-report])
(setv report (classification-report y-true y-pred
  :classes ["Cat" "Dog" "Bird"] :save "reports/classes.md"))
(print (get report "macro_avg"))
```

Calculated metrics include precision, recall, F1, support for each class,
overall accuracy, unweighted macro averages and support-weighted averages.
Classes with zero examples or predictions receive zero where division
would otherwise be undefined. Exports accept `.json` and `.md`.
No scikit-learn dependency is required.

### Raw cells and Jupyter code tags

```clojure
(require lispyn.macros [raw-cell cell-tags md])
(raw-cell "<!-- Markdown exporter hint -->")
(cell-tags "parameters" "hide-input")
(setv learning-rate 0.001)
```

Top-level `raw-cell` writes a **raw** Jupyter cell, preserving literal
strings without running them. `cell-tags` annotates the **next code cell**.
Tags must be unique, literal, nonempty Jupyter identifiers; the directive
must occur on its own line and cannot be left unattached. Jupyter
notebooks preserve these tags, and the macros are Hy runtime no-ops.
Repeat `(require ...)` in each generated code cell using macros.

### Comparing recorded experiments

```bash
lispyn compare runs/a runs/b --metric loss --split val \
  --output reports/compare.md --plot plots/compare.svg
```

`compare` reads `history.json` from each run without executing training.
Comparisons report the final metric and the best recorded value with its
1-based epoch. By default `loss` metrics are minimized; other metrics
are maximized. Override with `--mode min` or `--mode max`.
Export tables as `.md`, `.csv` or `.json`, and draw one figure containing
all runs with the same Docusaurus-dark/turquoise plotting themes.
The plotting dependency remains optional.

## v0.19 alpha: accumulating gradients, clipping and native LR schedules

Native PyTorch training can now accumulate gradients across microbatches
before each optimizer update, apply global gradient-norm clipping, and
step a learning-rate scheduler once per epoch. New options in both
`lispyn.workflow.train` and `lispyn.training.fit`:

- `accumulation_steps=1`: positive integer; `1` preserves the original behavior.
- `gradient_clip=None`: optional positive finite maximum global gradient norm.
- `scheduler=None`: native scheduler object in `fit`, or named
  `"cosine"`, `"step"`, `"plateau"`/PyTorch factory in `train`.
- `scheduler_kwargs=None`: additional native constructor options.

Gradient accumulation normalizes the mean loss across each group of
microbatches; the final partial group is scaled by its actual size.
The recorded losses/metrics still represent per-microbatch averages.
For variable microbatch sizes, the gradients are averaged by microbatch
rather than by individual sample. For exact per-sample weighting, handle
loss normalization in a custom PyTorch step instead.

```clojure
(require lispyn.macros [train-accumulating learning-rate-curve])
(setv run (train-accumulating model data 3
  :loss-fn criterion :validation validation
  :epochs 40 :lr 0.0003 :gradient-clip 1.0
  :scheduler "cosine" :scheduler-kwargs {"eta_min" 0.00001}
  :experiment-dir "runs/my-model"))
(learning-rate-curve run :save "runs/my-model/lr.svg")
```

Native `StepLR` and `CosineAnnealingLR` advance after each epoch.
`ReduceLROnPlateau` receives the tracked validation/training monitor.
`history['learning_rate']` records the **starting** optimizer learning rate
for each epoch and is included in the saved JSON and state dict history.
Schedulers are created after the optimizer, and can also be passed as
factories for custom PyTorch schedules.

Note: this release does not provide automatic mixed precision or
resume-from-checkpoint. For complex schedulers and multiple optimizer
parameter groups, the plotted learning rate is the first group's
learning rate. Use native PyTorch controls for additional requirements.


## Group-safe dataset splits and baseline-aware comparisons (0.20.0a1)

`train_validation_split(frame, stratify="label", seed=42)` divides a DataFrame
by class, keeping at least one record of each class in both partitions.
`train_validation_split(frame, group="subject_id", seed=42)` instead keeps all
rows belonging to each subject together. For arbitrary indexable PyTorch
datasets use `split_dataset(dataset, groups=subject_ids, seed=42)`, passing a
sequence with one group ID per sample. The number of groups must be at least
two; grouped and stratified modes cannot be combined. Group assignments are
reproducible for a fixed input and seed, but validation counts may differ
from the requested fraction because groups are indivisible. Neither split
requires scikit-learn.

`lispyn compare runs/baseline runs/candidate --baseline baseline` adds the
absolute `best_gain` and `final_gain` fields to Markdown, CSV, or JSON
reports. Gains are direction-aware: positive means improved relative to the
baseline whether the selected metric is minimized (loss) or maximized
(accuracy). The default comparison shape is unchanged.


## Train/validation/test and k-fold cross-validation (0.21.0a1)

**Author experiments in Hy (Lisp), with Lispyn macros as the main interface.**
Python/PyTorch implements the underlying data operations, and the resulting
`.hy` source can be compiled into an ordinary Jupyter notebook.

```hy
(require lispyn.macros [train-val-test cross-validation-folds])
(import torch)

; Small synthetic PyTorch dataset; no downloads.
(setv labels (+ (* [0] 15) (* [1] 15)))
(setv dataset (torch.utils.data.TensorDataset
               (torch.arange 30) (torch.tensor labels)))

; Reserve an untouched test set before cross-validation.
(setv splits (train-val-test dataset
              :labels labels
              :validation-fraction 0.2
              :test-fraction 0.1
              :seed 42))
(setv training (get splits 0))
(setv validation (get splits 1))
(setv testing (get splits 2))
(print "Train/val/test:" (len training) (len validation) (len testing))

; Labels must align with the training subset, not the original dataset.
(setv train-labels
  (lfor index (getattr training "indices") (get labels index)))
(for [[fold-train fold-val]
      (cross-validation-folds training
        :n-splits 5 :labels train-labels :seed 42)]
  ; Create a fresh model and optimizer for each fold here.
  (print "Fold sizes:" (len fold-train) (len fold-val)))

; Evaluate on testing only once, after model selection.
```

`train-val-test` and `cross-validation-folds` are real macros from
`lispyn.macros`, not new Python syntax. They expand to the PyTorch dataset
splitters at compile time. Use `lispyn build examples/cross_validation.hy`
to generate a notebook from the [Hy source](../examples/cross_validation.hy).

**Leakage-safe groups:** supply `:groups subject-ids` to `train-val-test`.
To run group-aware k-fold on its training subset, pass a matching sequence
of training-group IDs to `cross-validation-folds` rather than `:labels`.
Grouped and stratified modes are mutually exclusive. Each class needs at
least `n_splits` examples for stratified k-fold, and grouped k-fold needs
at least `n_splits` distinct groups. Three-way splitting requires at least
three class samples or groups; group boundaries may make sizes approximate.

**Tabular DataFrames** can also be used directly from Hy, without Python
source files:

```hy
(import lispyn.data [train-validation-test-split k-fold-split])
(setv splits (train-validation-test-split frame
              :stratify "label" :test-fraction 0.1 :seed 42))
(setv training-frame (get splits 0))
(for [[fold-train fold-val]
      (k-fold-split training-frame :n-splits 5 :stratify "label")]
  (print (len fold-train) (len fold-val)))
```

This is ordinary Hy syntax calling interoperable pandas functionality;
the PyTorch workflow above uses dedicated Lispyn macros.

For a quick summary of numerical fold metrics:

```bash
lispyn cv-summary 0.81 0.84 0.83 0.86 0.82 --metric accuracy --output folds.json
```

The summary reports mean, population standard deviation and range of the
*provided* fold scores. It does not train models or touch the held-out test set.
