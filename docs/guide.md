# Lispyn — Extended Guide

**Lisp macros for Python, data science, PyTorch and Jupyter.**

Lispyn is an experimental, open-source project originally created by **Eduardo J. Barrios**. AI makes code generation inexpensive; Lispyn explores a complementary idea: **create reusable abstractions for recurring ML experiments instead of regenerating the same code**.

Lispyn extends [Hy](https://hylang.org/) (a Lisp running on Python). It does **not** replace Python, hide PyTorch, or require developers to reimplement familiar ML APIs. Import Python modules and call their functions directly.

> Status: **v0.7.0a1 (experimental)**. No stability guarantees; not yet published on PyPI.

## Install from source

Requires Python 3.10+:

```bash
git clone https://github.com/edujbarrios/lispyn.git
cd lispyn
python -m pip install -e '.[notebook,data]'
```

Install PyTorch separately with the [official CPU/GPU instructions](https://pytorch.org/get-started/locally/). For Parquet loading, install `pyarrow` as well.

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
