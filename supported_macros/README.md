# Supported Lispyn macros

**The complete Hy/Lisp macro reference — 84 macros, with usage examples.**

Lispyn's API is **Hy macros**. They compose ordinary Python, pandas, PyTorch, Matplotlib and Jupyter; they do not replace those libraries. This file is the **single source of truth for supported macros**, grouped by AI notebook task (including research and production prototyping). The implementations live in [`src/lispyn/macros.hy`](../src/lispyn/macros.hy).

> Looking for a first experiment? Start with the [main README](../README.md) and the runnable [quickstart](../examples/quickstart.hy).

## How to read this catalog

In Hy, import macros with `require`:

```hy
(require lispyn.macros [with-seed train-until report-run])
```

The examples below show the public **Hy syntax**, not direct Python implementations. Names such as `model`, `dataset`, `frame`, `history`, `score-fold` and `train-loader` refer to objects you define in your experiment, unless an example creates them. Keyword arguments use Hy's `:option value` syntax, e.g. `:epochs 10`; hyphenated names compile to Python underscores. Most macros forward additional keywords to their named Python backend — consult the implementation for the complete parameter list.

**Important Jupyter rule:** Lispyn compiles notebook code cells **independently**. Use `(require lispyn.macros [...])` again in a new cell when that cell needs a macro. Top-level `md`, `pip-install`, `raw-cell`, `cell-tags` and `experiment-params` have special notebook-generation behavior and must occupy their own lines. `lispyn build` compiles trusted Hy without executing training; `lispyn execute` executes the generated Python notebook. Hy compile-time macros are not sandboxed, so only compile trusted source.

**Install:** `pip install lispyn` already includes Jupyter execution (`nbclient`, `ipykernel`) and Matplotlib plots. Install a compatible **PyTorch** build separately for training; the optional root [`requirements.txt`](../requirements.txt) provides an entire research environment from a cloned repository. Pandas (`lispyn[data]`), audio, vision and PEFT workflows are additional options. The older `[notebook,plots]` extras remain valid for compatibility but are no longer necessary.

## Index

| AI notebook task | Macros |
| --- | --- |
| [Notebook structure & parameters](#notebook-structure--parameters) | `md`, `markdown`, `raw-cell`, `cell-tags`, `pip-install`, `experiment-params` |
| [NumPy and pandas data profiling](#numpy-and-pandas-data-profiling) | `array-profile`, `table-profile` |
| [PyTorch training & inference](#pytorch-training--inference) | `with-seed`, `train-for`, `train-until`, `train-accumulating`, `with-eval`, `head-tune`, `full-tune` |
| [PyTorch controls & checkpoints](#pytorch-controls--checkpoints) | `with-no-grad`, `with-autocast`, `clip-gradients`, `freeze-module`, `unfreeze-module`, `parameter-count`, `move-to-device`, `save-model`, `load-model` |
| [Model and optimizer factories](#model-and-optimizer-factories) | `defmodel`, `defoptimizer` |
| [Custom Matplotlib plot generation](#custom-matplotlib-plot-generation) | `with-figure`, `defplot` |
| [Macro-first PyTorch composition](#macro-first-pytorch-composition) | `with-training`, `with-grad`, `with-rng-fork`, `optimizer-step`, `deftrain-step`, `collect-epochs` |
| [PyTorch diagnostics & inference](#pytorch-diagnostics--inference) | `gradient-report`, `assert-gradients`, `model-footprint`, `predict-batches`, `with-parameter-freeze`, `with-determinism`, `with-anomaly-detection` |
| [Runtime dependencies](#runtime-dependencies) | `ensure-package`, `use-package` |
| [AI dataset class balance](#ai-dataset-class-balance) | `label-balance`, `plot-label-balance`, `class-weights` |
| [Binary probability thresholds](#binary-probability-thresholds) | `threshold-report`, `plot-threshold-metrics` |
| [Visualization & classification](#visualization--classification) | `curve`, `defcurve`, `compare-curves`, `learning-rate-curve`, `confusion-matrix`, `defconfusion`, `classification-report` |
| [Dataset splits & cross-validation](#dataset-splits--cross-validation) | `train-val-test`, `cross-validation-folds`, `tabular-train-val-test`, `tabular-cross-validation-folds`, `repeated-cross-validation-folds`, `repeated-tabular-folds`, `cross-validate`, `cross-validate-table`, `save-cross-validation` |
| [Time-series experiments](#time-series-experiments) | `temporal-train-val-test`, `temporal-folds`, `temporal-table-split`, `temporal-table-folds`, `cross-validate-temporal`, `cross-validate-temporal-table` |
| [Input assertions for NumPy, pandas and PyTorch](#input-assertions-for-numpy-pandas-and-pytorch) | `assert-finite`, `assert-shape` |
| [Evaluation integrity guards](#evaluation-integrity-guards) | `assert-disjoint`, `assert-probabilities` |
| [Lightweight diagnostics](#lightweight-diagnostics) | `assert-close`, `with-timing` |
| [Reports, evidence & narratives](#reports-evidence--narratives) | `report-run`, `experiment-check`, `checklist-report`, `narrate`, `metric-table`, `experiment-leaderboard`, `compare-runs`, `plot-runs` |

## Notebook structure & parameters

### `md`
**Purpose:** Turn literal Markdown strings into a real Jupyter Markdown cell. It is a no-op when running a `.hy` file directly.

```hy
(require lispyn.macros [md])
(md "## Hypothesis" "Does increasing model capacity improve held-out accuracy?")
```

Call at top level, on its own lines. Dynamic text belongs in [`narrate`](#narrate), not `md`.

### `markdown`
**Purpose:** Longer-name alias for `md`; identical behavior and restrictions.

```hy
(require lispyn.macros [markdown])
(markdown "## Experimental setup" "Data and evaluation protocol.")
```

### `raw-cell`
**Purpose:** Add a raw (non-executable) Jupyter notebook cell. No-op in direct Hy execution.

```hy
(require lispyn.macros [raw-cell])
(raw-cell "LaTeX / publication-specific text kept as a raw cell.")
```

Only top-level literal strings; it does not render Markdown.

### `cell-tags`
**Purpose:** Assign Jupyter metadata tags to the **next code cell**.

```hy
(require lispyn.macros [cell-tags])
(cell-tags "hide-input")
(print "Executed, but the notebook may hide this source cell")
```

Valid literal tag names only; must immediately precede a code/parameters cell. Tags such as `hide-input` are interpreted by notebook frontends.

### `pip-install`
**Purpose:** Generate a separate Jupyter dependency cell. The selected named packages are checked/installed **when executed**, never when building the notebook.

```hy
(require lispyn.macros [pip-install])
(pip-install "matplotlib>=3.8" "pandas>=2")
```

Literal pip requirement strings only; no arbitrary shell commands or package URLs. Use only in trusted notebooks. **This macro does not install packages during `lispyn build`:** it emits a dedicated dependency cell that checks/installs the named library when executed with `lispyn execute`. For example, `(pip-install "scikit-learn>=1.5")` enables an optional library. A kernel restart may be required for compiled dependencies. For PyTorch CUDA/ROCm builds, prefer the official platform-specific installation.

### `experiment-params`
**Purpose:** Declare Hy variables once and compile a Jupyter code cell tagged `parameters`. Useful for repeatable variants.

```hy
(require lispyn.macros [experiment-params])
(experiment-params
  (setv seed 42)
  (setv epochs 10)
  (setv learning-rate 0.001))
```

Every child must be `(setv name value)`. Variables also work during direct Hy execution. Inspect with `lispyn parameters study.hy`; override with `lispyn build study.hy --set epochs=20` or `--params config.json`; create notebook grids with `lispyn sweep`.

## Input assertions for NumPy, pandas and PyTorch

Use these guards to **fail before training on bad input data**. They are lightweight and return the original object unchanged, so you can insert them into existing Hy experiments. Both use a descriptive optional `:name` and never print or store input values. NumPy is included with Matplotlib; pandas (`lispyn[data]`) and PyTorch (platform-specific installation) are optional.

### `assert-finite`
**Purpose:** Reject any NaN or infinity in numeric NumPy arrays/sequences or dense PyTorch tensors; return the original input unchanged if all values are finite.

```hy
(require lispyn.macros [assert-finite])
(import numpy :as np)
(setv features (np.array [1.0 2.0 3.0]))
(assert-finite features :name "training features")
```

Reports only the number of nonfinite elements on failure; for example, a sensor array containing a NaN fails before training rather than propagating an invalid loss. Rejects empty inputs, strings/object arrays and invalid names. PyTorch tensors keep their identity and gradient tracking. Does not mutate inputs or perform automatic cleaning.

### `assert-shape`
**Purpose:** Verify NumPy array, pandas DataFrame or PyTorch tensor dimensions, with `None` as a wildcard for unknown batch size; return the original object unchanged.

```hy
(require lispyn.macros [assert-shape])
(import numpy :as np)
(setv features (np.zeros [32 4]))
(assert-shape features [None 4] :name "training features")
```

Arguments: `values expected` and optional `:name`. Expected shape is a list or tuple of 1–8 nonnegative integers or `None`. Raises `AssertionError` on a mismatch, `TypeError` on non-shaped objects, and `ValueError` for invalid expected shapes. No reshaping or mutation occurs. See the [runnable direct/Notebook example](../examples/validate_inputs.hy).

## Evaluation integrity guards

These lightweight macros reject common mistakes that can make validation or test
metrics misleading. They do not automatically improve a trained model's accuracy.
Both are non-mutating, print no sample-level values, and work in direct Hy and
compiled Python notebooks.

### `assert-disjoint`
**Purpose:** Fail if two partitions share any stable sample identifiers; avoid
accidentally evaluating on training observations. Pass explicit IDs, not feature
values or dataset objects. Check all three pairs for train/validation/test.

```hy
(require lispyn.macros [assert-disjoint])
(assert-disjoint ["train-1" "train-2"] ["valid-1" "valid-2"])
(assert-disjoint training-ids test-ids :right-name "test")
```

Arguments: `left right`; optional `:left-name` and `:right-name` change error
labels. Inputs must be nonempty 1-D lists, tuples, NumPy arrays, pandas Series
or PyTorch tensors of string/integer IDs (not booleans). Repeated IDs within
one partition are permitted; overlaps are counted uniquely, and the failing
assertion never reveals the IDs. Strings and integers are distinct IDs.
Returns `True` if disjoint. This cannot detect duplicate examples that have
different IDs.

### `assert-probabilities`
**Purpose:** Reject malformed binary scores or multiclass probabilities before
computing evaluation metrics, rather than silently using logits, NaNs or
unnormalized rows.

```hy
(require lispyn.macros [assert-probabilities])
(assert-probabilities [0.05 0.85 0.40])
(assert-probabilities [[0.2 0.8] [0.6 0.4]] :atol 0.000001)
```

A 1-D vector of floating-point scores must be nonempty and within [0, 1].
A 2-D array must have at least two columns, contain only finite scores in
[0, 1], and have each row sum to 1 within `:atol` (default `1e-6`).
Use a larger tolerance for low-precision models if needed; the maximum
allowed tolerance is `0.01`. Supports Python float sequences, NumPy/pandas
arrays and dense floating-point PyTorch tensors, preserving input identity and
autograd state. Returns the input unchanged. No automatic softmax, sigmoid,
clipping or calibration is performed. See the
[runnable Hy example](../examples/evaluation_guards.hy).

## Lightweight diagnostics

These macros require only the default Lispyn installation and are useful in ordinary Hy scripts and generated notebooks. See [`examples/hello.hy`](../examples/hello.hy) for both in one runnable example.

### `assert-close`
**Purpose:** Assert that two finite numeric scalars are approximately equal; return `True` on success and raise a descriptive `AssertionError` on mismatch. Useful for smoke-testing calculations, expected metrics, and reproducibility checks.

```hy
(require lispyn.macros [assert-close])
(assert-close (+ 0.1 0.2) 0.3)
(assert-close 9.9 10 :abs-tol 0.2)
```

Optional `:rel-tol` (default `1e-9`) and `:abs-tol` (default `0.0`) are nonnegative. Booleans, NaNs, infinities and unsupported scalar types are rejected. This is for Python `int`/`float` scalars, not PyTorch tensors or vector comparisons.

### `with-timing`
**Purpose:** Measure wall-clock seconds for a Hy body, printing a labeled duration even if the body raises. Useful for comparing small steps without introducing a profiling dependency.

```hy
(require lispyn.macros [with-timing])
(with-timing "Build features"
  (setv features [(* x x) (for x (range 1000))])
  (print "Feature count:" (len features)))
```

The label must be a nonempty string. Timing includes runtime overhead and is not a rigorous benchmark.

## NumPy and pandas data profiling

Use these macros to check data quality **before training**. They display aggregate Markdown in Jupyter (or print it in direct Hy), and return JSON-compatible summaries. They never include individual rows or original array values. Save summaries with `:save "report.md"` or `:save "report.json"`. NumPy is included through Matplotlib; pandas requires `pip install "lispyn[data]"`.

### `array-profile`
**Purpose:** Summarize a 1-D or 2-D real numeric NumPy array: shape, dtype, finite values, NaNs/infinities and finite-only min/max/mean.

```hy
(require lispyn.macros [array-profile])
(import numpy :as np)
(setv summary
  (array-profile (np.array [1.0 2.0 (float "nan")])
    :name "Sensor values" :save "reports/array.json"))
(print (get summary "nan"))
```

Options: `:name` (label) and `:save` (`.md` or `.json`). Returns `kind`, `name`, `shape`, `dtype`, `total`, `finite`, `nan`, `infinite`, `minimum`, `maximum`, and `mean`. Excludes NaNs/infinities from statistics; reports `None` when no finite values exist. Rejects empty, complex, string/object, and higher-dimensional arrays. Does not mutate the array.

### `table-profile`
**Purpose:** Summarize pandas DataFrame rows, columns, missing cells, numeric infinities, duplicate rows, and per-column dtypes/missingness without printing data.

```hy
(require lispyn.macros [table-profile])
(import pandas :as pd)
(setv frame (pd.DataFrame {"site" ["a" "b"] "score" [1.0 None]}))
(setv summary
  (table-profile frame :name "Measurements"
    :max-columns 100 :save "reports/table.md"))
(print (get summary "missing_cells"))
```

Options: `:name`, `:save` (`.md` or `.json`), and `:max-columns` (default 100; max 500). Returns `kind`, `name`, `rows`, `columns`, `missing_cells`, `infinite_cells`, `duplicate_rows`, and `fields` (column label, dtype, missing count, infinity count). Requires unique column labels and does not mutate the DataFrame. **Column names can be sensitive**, so review exported reports before sharing.

[Try both macros in a runnable Hy example](../examples/data_profiles.hy).

## PyTorch training & inference

### `with-seed`
**Purpose:** Seed Python, NumPy (when present) and PyTorch before evaluating a Hy body.

```hy
(require lispyn.macros [with-seed])
(with-seed 42
  (setv model (torch.nn.Linear 2 2)))
```

Seed before initializing a model/dataloader; deterministic results still depend on hardware and kernels.

### `train-for`
**Purpose:** Call the lower-level native PyTorch training loop when you already manage the dataloader, optimizer and loss function.

```hy
(require lispyn.macros [train-for])
(setv history (train-for model train-loader optimizer loss-fn 10))
```

Arguments: `model loader optimizer criterion epochs`. Returns a metric history mapping from `lispyn.training.fit`. For automatic loader/optimizer setup, use `train-until` or Python `lispyn.workflow.train`.

### `train-until`
**Purpose:** Train a native PyTorch model for up to `:epochs` epochs with early stopping after `patience` epochs without sufficient improvement.

```hy
(require lispyn.macros [train-until])
(setv run
  (train-until model training 3
    :validation validation
    :loss-fn (torch.nn.CrossEntropyLoss)
    :epochs 25 :batch-size 16 :lr 0.001 :seed 42))
```

Arguments: `model data patience` and optional `train` workflow kwargs (`:device`, `:metrics`, `:monitor`, `:experiment-dir`, `:save-best`, etc.). Returns a `Run` with `.history`, `.model`, `.optimizer` and `.stopped_early`. Data can be a PyTorch Dataset/DataLoader or compatible batches.

### `train-accumulating`
**Purpose:** Train using gradient accumulation over `steps` microbatches per optimizer update.

```hy
(require lispyn.macros [train-accumulating])
(setv run
  (train-accumulating model training 4
    :loss-fn loss-fn :epochs 5 :gradient-clip 1.0))
```

Arguments: `model data steps`. Forwards other options to `lispyn.workflow.train`, including `:scheduler`, `:validation`, `:lr`, `:batch-size`. Useful for memory-constrained experiments.

### `with-eval`
**Purpose:** Temporarily use PyTorch evaluation and inference mode; restore **all child-module train/eval flags** even on error.

```hy
(require lispyn.macros [with-eval])
(setv logits (with-eval model (model features)))
```

Arguments: `model` plus Hy body. Expands to `torch.inference_mode()` and `model.eval()` with hygienic temporaries and `try/finally`. Each child module recovers its own prior training mode, including deliberately frozen/evaluated layers.

### `head-tune`
**Purpose:** Freeze all but a named model head (e.g. ResNet `fc`) and train that head.

```hy
(require lispyn.macros [head-tune])
(setv run
  (head-tune model training "fc"
    :validation validation :loss-fn loss-fn :epochs 5))
```

Arguments: `model data head` plus normal `train` options. The head name must match a model parameter prefix; `lispyn.workflow.train` creates a fresh optimizer for trainable parameters.

### `full-tune`
**Purpose:** Unfreeze all model parameters and fine-tune with a newly created optimizer.

```hy
(require lispyn.macros [full-tune])
(setv run
  (full-tune model training
    :validation validation :loss-fn loss-fn :epochs 10 :lr 0.0001))
```

Arguments: `model data` plus normal `train` options. Often follows `head-tune` in transfer learning.

## PyTorch controls & checkpoints

These optional PyTorch macros help when you need **your own training loop** rather than
`train-until` / `train-accumulating`. They call native PyTorch APIs, require a
separately installed torch build, and work in both direct Hy and compiled Jupyter.
[Run the CPU-only example](../examples/pytorch_controls.hy) to see them together.

### `with-no-grad`
**Purpose:** Disable autograd for a Hy body and restore the previous grad-enabled state, including on exceptions. Unlike `with-eval`, this does not switch a model into evaluation mode.

~~~hy
(require lispyn.macros [with-no-grad])
(setv scores (with-no-grad (model features)))
~~~

Accepts one or more Hy forms; returns the last expression. Use `with-eval` when you also need `model.eval()` and inference-mode semantics.

### `with-autocast`
**Purpose:** Run forward computations inside native PyTorch autocast with an **explicit** device type and floating-point dtype.

~~~hy
(require lispyn.macros [with-autocast])
(setv logits (with-autocast "cpu" torch.bfloat16 (model features)))
~~~

Arguments: `device-type dtype` and a Hy body. Return value is the body's final expression. For CUDA, use `"cuda"` and a supported dtype on a compatible device. Keep backward outside the autocast block; when training with CUDA float16, follow PyTorch's `torch.amp.GradScaler` guidance to avoid underflow. Autocast is not the same as `no_grad`.

### `clip-gradients`
**Purpose:** Clip an existing model's global gradient norm **in place**, rejecting NaN or infinite norms.

~~~hy
(require lispyn.macros [clip-gradients])
(.backward loss)
(setv before-clip (clip-gradients model 1.0 :norm-type 2.0))
(.step optimizer)
~~~

Arguments: `model max-norm`, optional `:norm-type` (positive float or infinity). Returns the **pre-clipping global norm** as a float. Fails if there are no gradients or the limit is invalid. When using `GradScaler`, unscale the optimizer's gradients first. This is a manual-loop helper; `train-accumulating` also accepts `:gradient-clip`.

### `freeze-module`
**Purpose:** Turn off `requires_grad` for every parameter of a torch module, returning the same module.

~~~hy
(require lispyn.macros [freeze-module])
(freeze-module model.encoder)
~~~

Does **not** change train/eval state. BatchNorm running statistics can still update in training mode; call `.eval()` separately if needed. Recreate/configure optimizers after changing trainable parameters.

### `unfreeze-module`
**Purpose:** Turn `requires_grad` back on for every parameter of a module, returning that module.

~~~hy
(require lispyn.macros [unfreeze-module])
(unfreeze-module model.encoder)
~~~

This does not change the module's train/eval state; create an optimizer that includes the desired parameters.

### `parameter-count`
**Purpose:** Inspect native parameter element counts, including frozen and trainable subsets.

~~~hy
(require lispyn.macros [parameter-count])
(setv counts (parameter-count model))
(print (get counts "trainable"))
~~~

Returns `{"total": int, "trainable": int, "frozen": int}`, counting shared parameters once through `model.parameters()`. Does not calculate activation memory or FLOPs.

### `move-to-device`
**Purpose:** Move PyTorch tensors inside nested dictionaries, lists, tuples or namedtuples to the specified torch device, leaving non-tensor values untouched.

~~~hy
(require lispyn.macros [move-to-device])
(setv batch (move-to-device {"x" features "labels" targets} "cpu"))
~~~

Arguments: `batch device`, optional `:non-blocking True` (default `False`). Returns a new collection structure, preserving ordinary list/tuple/namedtuple shapes; mappings become dictionaries. A CPU-only install cannot move to CUDA. Nonblocking transfer requires appropriate hardware and memory conditions.

### `save-model`
**Purpose:** Save native PyTorch state dictionaries for a model and optionally optimizer and metric history, not the model's Python class.

~~~hy
(require lispyn.macros [save-model])
(save-model model "runs/baseline.pt" :optimizer optimizer
  :history {"train_loss" [0.5 0.3]})
~~~

Returns the destination `Path` and creates missing parent directories. Keyword options: `:optimizer`, `:history`. For higher-level training use `:save-best` and `:save-last` options instead.

### `load-model`
**Purpose:** Restore a **trusted** state-dict checkpoint with `weights_only=True` and optionally load optimizer state.

~~~hy
(require lispyn.macros [load-model])
(setv history (load-model model "runs/baseline.pt" :optimizer optimizer
  :map-location "cpu"))
~~~

Returns recorded history, or `None`; restores the provided model **in place**. Options: `:optimizer`, `:map-location` (defaults to CPU). Only load files you trust; `weights_only=True` reduces but does not eliminate untrusted-file risks. Ensure model architecture and optimizer are compatible.

## Model and optimizer factories

These macros generate ordinary Python functions rather than routing model or
optimizer construction through a Lispyn runtime. Install PyTorch independently
using the CPU or accelerator build appropriate for your machine.

### `defmodel`
**Purpose:** Generate a parameterized factory for a fresh native
`torch.nn.Sequential` network, with layers written directly in Hy.

```hy
(require lispyn.macros [defmodel])
(import torch)
(defmodel make-mlp [features hidden]
  (torch.nn.Linear features hidden)
  (torch.nn.ReLU)
  (torch.nn.Linear hidden 1))
(setv model (make-mlp 2 8))
```

The generated `make-mlp` is a plain Python function; each invocation
instantiates new layers and weights. Its arguments are ordinary Hy function
parameters, and layers are evaluated in the order supplied when called.
A valid parameter list and at least one layer are required. Supports
sequential architectures only; use standard Python `torch.nn.Module`
for branching graphs, weight sharing, or custom forward passes.

### `defoptimizer`
**Purpose:** Generate an optimizer factory that consistently binds a model's
parameters and learning rate, while leaving the optimizer class explicit.

```hy
(require lispyn.macros [defoptimizer])
(import torch)
(defoptimizer make-adam [model rate] torch.optim.Adam :weight-decay 0.01)
(setv optimizer (make-adam model 0.001))
```

The signature must be two symbols `[model learning-rate]`, and the named
constructor must accept `model.parameters()`, `:lr` and any additional
keyword options (such as `:weight-decay`). Both the model and rate are
evaluated once by the generated function. This does not choose a device,
optimizer, loss, or learning rate automatically, or imply higher accuracy.

## Custom Matplotlib plot generation

These source-generating macros make arbitrary Matplotlib figures from data
while remaining inspectable as standard Python. They complement the
specialized `curve` and `defcurve` macros for training histories.
Matplotlib is included with the default Lispyn installation.

### `with-figure`
**Purpose:** Create a native Matplotlib figure and axes with explicit
user-provided names, evaluate plotting forms, and return the figure.

```hy
(require lispyn.macros [with-figure])
(setv figure (with-figure [fig ax]
  (.plot ax [0 1 2] [0.8 0.4 0.2] :label "Loss")
  (.set_title ax "Validation loss")
  (.legend ax)))
(.savefig figure "validation-loss.png")
```

Expands to `pyplot.subplots()`, the user body and `figure.tight_layout()`.
The figure remains open on success so you can display, save, or modify it.
If the body raises, the generated `except BaseException` closes that
figure and re-raises. Close successful figures yourself with
`matplotlib.pyplot.close(figure)`. Requires exactly `[figure axes]`
symbols and one or more expressions.

### `defplot`
**Purpose:** Generate a reusable Python plotting function that creates a
fresh figure for each call, returning it for optional export.

```hy
(require lispyn.macros [defplot])
(defplot show-loss [losses] [fig ax]
  (.plot ax (range (len losses)) losses :marker "o")
  (.set_title ax "Training loss"))
(setv figure (show-loss [0.8 0.4 0.2]))
(.savefig figure "loss.png")
```

Use `(defplot name [parameters] [figure axes] body...)`. All plotting
operations are native Matplotlib. Just as in `with-figure`, figures
remain open on success and are closed automatically on plotting errors.
The function creates a new figure on each call; close it after use.
See [the runnable PyTorch and plotting showcase](../examples/showcase.hy).

## Macro-first PyTorch composition

The six forms in this section are **real Hy macros**: they generate
ordinary Python `def`, `for`, `try/finally` and native PyTorch calls at
compile time. They require PyTorch at **runtime**, but introduce no new
Lispyn Python backend. See [how to write your own macros](../MACRO_AUTHORING.md)
and the [CPU-only, executable example](../examples/macro_first_training.hy).

### `with-training`
**Purpose:** Enable `model.train()` and autograd temporarily for BODY,
restoring each child module's original training flag even after exceptions.

```hy
(require lispyn.macros [with-training])
(with-training model
  (setv train-loss ((torch.nn.MSELoss) (model features) labels)))
```

Evaluates the model expression once. Does not implicitly call
`backward()`, mutate optimizer configuration or restore changed weights.
If a module intentionally has mixed train/eval child states, all flags
are preserved. Entering this scope overrides an enclosing `no_grad()`
only until BODY finishes.

### `with-grad`
**Purpose:** Temporarily enable native PyTorch autograd inside a
`torch.no_grad()`-style scope, restoring the previous grad state.

```hy
(require lispyn.macros [with-grad])
(with-grad
  (setv tracked (model features)))
```

Equivalent to a scoped native `torch.enable_grad()`; it does **not**
override PyTorch's stronger `inference_mode()` semantics or make
inference tensors automatically differentiable.

### `with-rng-fork`
**Purpose:** Temporarily fork and restore the **CPU** PyTorch random number
generator without leaking seeded draws into subsequent experiments.

```hy
(require lispyn.macros [with-rng-fork])
(setv sampled (with-rng-fork
  (torch.manual_seed 42)
  (torch.randn [3])))
```

Expands to `torch.random.fork_rng(devices=[])`. It does **not** isolate
Python's `random`, NumPy's generators or CUDA device RNG states; for
a broader initial seed use `with-seed` and platform-appropriate workers.

### `optimizer-step`
**Purpose:** Expand one explicit native PyTorch optimizer update: zero
gradients, compute scalar loss, backward, then optimizer step.

```hy
(require lispyn.macros [optimizer-step])
(setv detached-loss (optimizer-step optimizer
  (criterion (model features) labels)))
```

The optimizer expression and loss expression are each evaluated **once**.
Returns the detached loss tensor, so convert with `.item()` when you
need a Python scalar. Expects a scalar differentiable loss. Does not
add gradient clipping, AMP, gradient accumulation or loss scaling;
configure those separately for your experiment.

### `deftrain-step`
**Purpose:** Generate an ordinary Python function for supervised
PyTorch training rather than routing through a Lispyn training engine.

```hy
(require lispyn.macros [deftrain-step])
(deftrain-step train-one model optimizer criterion)
(setv loss-as-float (train-one features labels))
```

The generated `train-one(inputs, labels)` function evaluates the supplied
model, optimizer and loss callable once per call, clears gradients, runs
forward/loss/backward/step, and returns a Python float. The model remains
in its preexisting train/eval mode; use `with-training` when needed.
This intentionally covers only the simple `criterion(model(inputs), labels)`
pattern. For custom batch mappings, accumulation or mixed precision,
use native PyTorch or the separately documented advanced Lispyn macros.

### `collect-epochs`
**Purpose:** Expand a Python loop that collects BODY's final result into a
fresh list for each iteration; useful for experiment history tracking.

```hy
(require lispyn.macros [collect-epochs])
(setv losses (collect-epochs epoch 5
  (print "Epoch" epoch)
  (train-one features labels)))
```

The user-provided `epoch` variable starts at zero and is bound once per
iteration; `count` is used as `range(count)`. The generated result list
uses a hygienic `hy.gensym` temporary that cannot overwrite the caller's
variables. Negative counts produce an empty list as in Python's `range`;
the macro does not validate training schedules or stop on its own.

## PyTorch diagnostics & inference

All seven macros in this section are optional PyTorch helpers. They complement,
rather than replace, native forward/backward loops and Lispyn's established
training workflow. See the [self-contained CPU example](../examples/torch_diagnostics.hy).
For generated notebooks, repeat `require` in **each code cell** using a macro.

### `gradient-report`
**Purpose:** Summarize dense gradients after `backward()` without modifying them or exposing tensor contents.

~~~hy
(require lispyn.macros [gradient-report])
(setv info (gradient-report model))
(print (get info "global_l2_norm"))
~~~

Returns a JSON-compatible mapping: `parameters`, `parameters_with_grad`,
`gradient_elements`, `nonfinite_elements`, `all_finite`,
`global_l2_norm` and `max_absolute`. Norm and maximum are `None` if
gradients are absent or not finite. Sparse gradients are rejected explicitly.
It synchronizes tensor statistics to the CPU; avoid calling it per microbatch
unless needed for debugging.

### `assert-gradients`
**Purpose:** Reject missing or nonfinite dense gradients before calling `optimizer.step()`.

~~~hy
(require lispyn.macros [assert-gradients])
(.backward loss)
(setv checked (assert-gradients model))
~~~

Returns the same aggregate summary as `gradient-report` if checks pass.
By default requires at least one parameter gradient; set `:require-any False`
to accept frozen models with no gradients. It does not clip or clear gradients,
and it does not replace checking for unstable parameter updates.

### `model-footprint`
**Purpose:** Estimate nominal storage used by unique PyTorch model parameter and buffer elements.

~~~hy
(require lispyn.macros [model-footprint])
(setv memory (model-footprint model))
(print (get memory "total_mib"))
~~~

Returns `parameter_elements`, `buffer_elements`, `parameter_bytes`,
`buffer_bytes`, `total_bytes` and `total_mib`. Includes both trainable
and frozen parameters. This **does not** estimate activation, gradient,
optimizer, allocator or peak GPU memory, nor guarantee physical allocation
for meta tensors or shared underlying storage.

### `predict-batches`
**Purpose:** Gather a bounded sequence of model logits into a CPU tensor while
using native `torch.inference_mode()` and restoring the original train/eval state.

~~~hy
(require lispyn.macros [predict-batches])
(setv logits (predict-batches model validation-loader
  :device "cpu" :max-batches 5))
~~~

Takes `model batches` and options `:device` (default `"cpu"`),
`:max-batches` (optional positive integer), and `:input-key` (required
when batches are dictionaries). Accepts a Tensor batch, a one/two-element
tuple/list where the first item is the input (e.g. `(inputs, labels)`), or
a mapping with a named input field. The input can be a nested mapping or
tuple to support `model(**inputs)` / `model(*inputs)`. Accepts model output
as a Tensor, an object with `.logits`, or a mapping with `"logits"`.
All outputs must be batched, tensor-valued and concatenation-compatible.

**Important:** Predictions are kept in CPU memory; `max-batches` limits
work, not tensor size. Model weights are moved to the specified device and
**stay there**, while its train/eval mode is restored even on error. This
helper does not collect labels, calculate metrics, persist data or make
model quality claims. For huge datasets, use a streaming native PyTorch loop.

### `with-parameter-freeze`
**Purpose:** Expand a `try/finally` scope that temporarily disables gradient
flags for every parameter in a module and restores each flag individually,
including after an exception, without delegating to a Python context helper.

~~~hy
(require lispyn.macros [with-parameter-freeze])
(with-parameter-freeze model.encoder
  (setv frozen-output (model.encoder features)))
~~~

Returns the last body expression. Does not change train/eval mode; BatchNorm
running statistics and dropout behavior can still change. Use `freeze-module`
instead for a persistent change, and configure optimizers explicitly.

### `with-determinism`
**Purpose:** Generate native PyTorch deterministic-algorithm scope code with
`try/finally` restoring the previous policy when the Hy body exits.

~~~hy
(require lispyn.macros [with-determinism])
(with-determinism
  (setv output (model features)))
~~~

Returns the last body expression. This does **not** seed RNGs, configure
DataLoader workers, control third-party operations, or set required CUDA
environment variables. Unsupported nondeterministic operations may raise
errors; see PyTorch's platform guidance. For warn-only behavior use the Python
`lispyn.torch_diagnostics.deterministic_algorithms(warn_only=True)`
context manager.

### `with-anomaly-detection`
**Purpose:** Enable PyTorch backward anomaly tracing around a short Hy body
to locate operations responsible for NaN gradients.

~~~hy
(require lispyn.macros [with-anomaly-detection])
(with-anomaly-detection
  (setv loss ((torch.nn.MSELoss) (model features) target))
  (.backward loss))
~~~

Returns the last body expression. Keep both forward and backward within the
context when you want useful forward traces. This adds overhead and may emit
debugging details, so use it during **local, trusted debugging only**, not
production training or on private data you cannot safely log.

## Runtime dependencies

### `ensure-package`
**Purpose:** Check a named Python distribution/module at runtime and optionally install it into the current interpreter.

```hy
(require lispyn.macros [ensure-package])
(ensure-package "numpy>=2" :install False)
```

`:install False` raises if missing (no downloads). `:import-name "sklearn"` handles distributions whose import name differs. Execution may modify an environment if installation is enabled; building never installs.

### `use-package`
**Purpose:** Ensure a package/module and return the imported Python module for further Hy use.

```hy
(require lispyn.macros [use-package])
(setv math (use-package "math"))
(print (math.sqrt 16))
```

Also accepts `:import-name` and `:install False`. Like `ensure-package`, performs dependency checks only at runtime.

## AI dataset class balance

These macros help **inspect classification datasets before training**, whether you use plain Python lists, NumPy 1-D arrays, pandas Series, or optional PyTorch 1-D tensors. They count only labels; no model training or data downloads occur. Labels themselves appear in summaries and plots, so check them before sharing notebooks.

### `label-balance`
**Purpose:** Count examples in each observed class, display a Markdown table with percentages and an imbalance ratio, and return a JSON-compatible summary.

```hy
(require lispyn.macros [label-balance])
(setv labels ["cat" "cat" "cat" "dog" "bird"])
(setv summary (label-balance labels
  :title "Training classes" :save "reports/classes.json"))
(print (get summary "majority_share"))
```

Returns `title`, `total`, `num_classes`, `majority_share`, `imbalance_ratio`, and `classes` entries (`label`, `count`, `share`). First-seen order is preserved. `:save` supports `.json` and `.md`. `:max-classes` defaults to 30, maximum 50. Rejects empty, non-1D or ambiguous labels and NaNs/infinities rather than silently removing observations. This is descriptive, **not** a statistical balance test or a substitute for proper train/test splitting.

### `class-weights`
**Purpose:** Produce balanced inverse-frequency class weights for imbalanced classification losses, without an optional PyTorch dependency.

```hy
(require lispyn.macros [class-weights])
(setv labels ["cat" "cat" "cat" "dog" "dog" "bird"])
(setv result (class-weights labels :class-order ["bird" "cat" "dog"]))
(print (get result "weights"))  ; [2.0, 0.666..., 1.0]
```

Returns JSON-compatible `class_order`, `counts`, `weights`, `total`, and `num_classes`. Uses `n_samples / (n_classes * class_count)` (the balanced inverse-frequency formula). If `:class-order` is omitted, labels are in first-seen order. **For PyTorch `CrossEntropyLoss`, provide `:class-order` exactly matching your target encoding `0..C-1`**, then convert `result["weights"]` to a device/dtype-compatible torch tensor. Each observed class must appear exactly once in the explicit order; unknown/duplicate/missing classes are rejected. This also distinguishes numeric labels from strings with the same display text. `:max-classes` defaults to 30 (up to 50), and inputs must contain at least one example of every class. Weights do not replace a representative validation split or guarantee better accuracy. See [the runnable example](../examples/label_balance.hy).

### `plot-label-balance`
**Purpose:** Generate a labeled Matplotlib bar chart of observed class counts without writing a plotting loop.

```hy
(require lispyn.macros [plot-label-balance])
(plot-label-balance ["cat" "cat" "dog" "bird"]
  :title "AI training labels" :theme "paper"
  :save "figures/labels.png")
```

Returns a Matplotlib `Figure` (for further customization or notebook display). Supports `:title`, `:theme` (`paper`, `clean`, `dark`, `docusaurus-dark`), `:max-classes`, `:figsize`, `:dpi` and `:save` (`.png` or `.svg`). Default Lispyn installations include Matplotlib. [Run the end-to-end Hy demonstration](../examples/label_balance.hy).

## Binary probability thresholds

Evaluate the trade-offs of a **binary classifier's positive-class probabilities** using held-out **validation** labels. These macros require neither scikit-learn nor PyTorch. They do not run inference or train a model. Do **not** optimize on the held-out test set: select your threshold on validation data and evaluate once on untouched test data.

### `threshold-report`
**Purpose:** Display aggregate precision, recall, specificity, F1, balanced accuracy and confusion counts at candidate cutoffs; optionally save a Markdown/JSON report.

```hy
(require lispyn.macros [threshold-report])
(setv actual [0 0 1 1 1 0 1 0])
(setv scores [0.1 0.35 0.4 0.8 0.7 0.9 0.55 0.2])
(setv result (threshold-report actual scores
  :thresholds [0.3 0.5 0.75] :optimize "f1"
  :save "reports/thresholds.json"))
(print (get result "best_threshold"))
```

Predicted positive means `score >= threshold`; `actual` contains integer 0/1 or Boolean labels and `scores` contains finite probabilities in [0,1] (positive-class probabilities, **not raw logits**). Lengths must match, be nonempty 1-D sequences and include both classes. Optional `:thresholds` accepts 1–101 distinct candidate cutoffs in [0,1] (default: 0.05 through 0.95 in steps of 0.05), sorted in the report. `:optimize` is `"f1"` (default) or `"balanced_accuracy"`. If metrics tie, choose the cutoff nearest 0.5, then the smaller cutoff. Metrics with zero prediction denominators are 0, and the search considers only the supplied grid. Returns JSON-compatible `best_threshold`, `best_metrics`, `rows`, class counts and objective. `:save` supports `.md`/`.json`; exports contain aggregate counts and metrics only, not individual labels/scores. This is **descriptive validation analysis**, not calibrated confidence or a guarantee of test performance.

### `plot-threshold-metrics`
**Purpose:** Plot the precision, recall, F1 and balanced accuracy trade-offs across candidate thresholds and mark the validation-selected cutoff.

```hy
(require lispyn.macros [plot-threshold-metrics])
(plot-threshold-metrics [0 1 0 1] [0.1 0.8 0.4 0.9]
  :thresholds [0.25 0.5 0.75] :theme "paper"
  :save "figures/thresholds.png")
```

Returns a Matplotlib `Figure`; supports the same `:thresholds` and `:optimize` arguments as `threshold-report`, `:title`, standard themes (`paper`, `clean`, `dark`, `docusaurus-dark`), `:figsize`, `:dpi` and `:save` (`.png`/`.svg`). Default installation includes Matplotlib. [Run the synthetic Hy/Jupyter demo](../examples/binary_thresholds.hy).

## Visualization & classification

### `curve`
**Purpose:** Plot training/validation metrics from a `Run`, a history dict or saved history.

```hy
(require lispyn.macros [curve])
(curve run :metric "loss" :theme "paper" :save "figures/loss.png")
```

Plotting options include `:theme`, `:palette`, `:title`, `:subtitle`, `:figsize`, `:dpi` and label/colors. Requires the `plots` extra.

### `defcurve`
**Purpose:** Define a reusable Matplotlib curve style as a callable Hy binding.

```hy
(require lispyn.macros [defcurve])
(defcurve journal-style :theme "paper" :line-width 2.5)
(journal-style run :metric "accuracy" :save "figures/accuracy.svg")
```

The callable forwards later overrides to `lispyn.plots.plot_history`. Great for consistent figures across several experiments.

### `compare-curves`
**Purpose:** Draw a common metric from multiple named histories in one chart.

```hy
(require lispyn.macros [compare-curves])
(compare-curves {"baseline" baseline-history "tuned" tuned-history}
  :metric "loss" :split "val" :save "figures/comparison.png")
```

Accepts a name-to-history/path mapping; each history needs the selected metric (e.g. `val_loss`). This plots results, not a significance test.

### `learning-rate-curve`
**Purpose:** Visualize the learning-rate history recorded during PyTorch training.

```hy
(require lispyn.macros [learning-rate-curve])
(learning-rate-curve run.history :save "figures/learning_rate.png")
```

Only useful when the history includes a recorded `learning_rate` series; works with native scheduling.

### `confusion-matrix`
**Purpose:** Plot a classification confusion matrix with native label lists and Matplotlib.

```hy
(require lispyn.macros [confusion-matrix])
(confusion-matrix actual predicted
  :classes ["cat" "dog"] :normalize "true"
  :theme "docusaurus-dark" :save "figures/confusion.png")
```

Actual/predicted arrays must have equal length. `:normalize "true"` normalizes by actual-class totals.

### `defconfusion`
**Purpose:** Create a reusable confusion-matrix plotter with default style/options.

```hy
(require lispyn.macros [defconfusion])
(defconfusion journal-matrix :theme "paper" :figsize [7 5])
(journal-matrix actual predicted :save "figures/confusion.svg")
```

Use for repeatable classification visuals in a paper or research group.

### `classification-report`
**Purpose:** Calculate per-class precision, recall, F1, support and aggregate accuracy without scikit-learn.

```hy
(require lispyn.macros [classification-report])
(setv report
  (classification-report actual predicted
    :classes ["cat" "dog"] :save "reports/classification.md"))
(print (get report "accuracy"))
```

Saves `.md` or `.json`; returns a dict with `accuracy`, `per_class`, `macro_avg` and `weighted_avg`.

## Dataset splits & cross-validation

Use **a true held-out test set** for the final assessment. Do not tune hyperparameters using test data. If observations share a subject/session, prefer grouped splitting. All fold scorers should create a *fresh* model and optimizer per fold.

### `train-val-test`
**Purpose:** Split an indexable PyTorch Dataset into three disjoint native `Subset` objects.

```hy
(require lispyn.macros [train-val-test])
(setv [training validation held-out]
  (train-val-test dataset :validation-fraction 0.2
    :test-fraction 0.2 :seed 42))
```

Also supports `:labels`/`:stratify True` for class-balanced splits or `:groups` for subject isolation (not both together). No model training occurs.

### `cross-validation-folds`
**Purpose:** Yield non-overlapping PyTorch (training, validation) subsets for k-fold cross-validation.

```hy
(require lispyn.macros [cross-validation-folds])
(for [[training validation] (cross-validation-folds dataset :n-splits 5 :seed 42)]
  (print (len training) (len validation)))
```

Accepts `:labels`, `:stratify True` or `:groups` as appropriate.

### `tabular-train-val-test`
**Purpose:** Split a pandas DataFrame into disjoint training, validation and held-out test frames.

```hy
(require lispyn.macros [tabular-train-val-test])
(setv [training validation held-out]
  (tabular-train-val-test frame :seed 42 :stratify "label"))
```

Use `:group "subject_id"` instead of `:stratify` when repeated subjects must stay together.

### `tabular-cross-validation-folds`
**Purpose:** Yield pandas (training, validation) DataFrames for each fold.

```hy
(require lispyn.macros [tabular-cross-validation-folds])
(setv folds (list (tabular-cross-validation-folds frame
  :n-splits 5 :group "subject_id")))
```

Preserves group boundaries or stratifies via a column name.

### `repeated-cross-validation-folds`
**Purpose:** Repeat shuffled PyTorch k-fold splits for multiple random rounds.

```hy
(require lispyn.macros [repeated-cross-validation-folds])
(setv folds (list (repeated-cross-validation-folds dataset
  :n-splits 5 :n-repeats 3 :seed 42)))
```

Each yielded tuple is `(repeat, fold, train_subset, valid_subset)`. Repeat/fold indices start at zero.

### `repeated-tabular-folds`
**Purpose:** Yield repeated k-fold splits for pandas DataFrames.

```hy
(require lispyn.macros [repeated-tabular-folds])
(setv folds (list (repeated-tabular-folds frame
  :n-splits 4 :n-repeats 2 :group "subject_id")))
```

Returns `(repeat, fold, train_frame, valid_frame)` tuples.

### `cross-validate`
**Purpose:** Apply a Hy scoring function to PyTorch folds, returning numerical summaries.

```hy
(require lispyn.macros [cross-validate])
; score-fold creates a fresh model and returns one finite score.
(setv report (cross-validate dataset score-fold
  :n-splits 5 :n-repeats 2 :metric "val_loss"))
```

`score-fold(training, validation, fold, repeat)` receives **zero-based** indices and must return a finite real number. Optional `:groups`, `:labels`, `:seed`, `:output "cv.json"`.

### `cross-validate-table`
**Purpose:** Apply a scoring callback to pandas cross-validation folds.

```hy
(require lispyn.macros [cross-validate-table])
(setv report (cross-validate-table frame score-fold
  :group "subject_id" :n-splits 5 :metric "val_accuracy"))
```

The callback receives `(training_frame, validation_frame, fold, repeat)`. A fresh model per fold remains the caller's responsibility.

### `save-cross-validation`
**Purpose:** Persist a cross-validation score summary as privacy-minimal JSON.

```hy
(require lispyn.macros [save-cross-validation])
(save-cross-validation report "reports/cross_validation.json")
```

The report contains aggregate scores, sizes and folds, not source samples or model weights.

## Time-series experiments

Temporal folds are **chronological**: training always precedes validation. `:gap` is a **number of records**, not time duration. Use these instead of shuffled folds for time-dependent observations.

### `temporal-train-val-test`
**Purpose:** Chronologically split a PyTorch Dataset into three Subsets, separated by optional embargo gaps.

```hy
(require lispyn.macros [temporal-train-val-test])
(setv [training validation held-out]
  (temporal-train-val-test time-ordered-dataset
    :validation-fraction 0.2 :test-fraction 0.2 :gap 2))
```

Input must already be time-ordered. Gap rows are excluded from all three subsets.

### `temporal-folds`
**Purpose:** Yield expanding or rolling train/validation PyTorch windows.

```hy
(require lispyn.macros [temporal-folds])
(setv windows (list (temporal-folds time-ordered-dataset
  :n-splits 4 :test-size 8 :gap 2 :min-train-size 12)))
```

Specify `:max-train-size` for a **rolling** instead of expanding training window.

### `temporal-table-split`
**Purpose:** Chronologically split a pandas DataFrame into train/validation/test.

```hy
(require lispyn.macros [temporal-table-split])
(setv [training validation held-out]
  (temporal-table-split frame :time-column "timestamp" :gap 1))
```

The timestamp column is sorted ascending and must be unique and nonmissing. The input is not mutated.

### `temporal-table-folds`
**Purpose:** Yield pandas expanding/rolling walk-forward windows.

```hy
(require lispyn.macros [temporal-table-folds])
(setv windows (list (temporal-table-folds frame
  :time-column "timestamp" :n-splits 5 :gap 2)))
```

No shuffle; `:max-train-size` caps the training window.

### `cross-validate-temporal`
**Purpose:** Score PyTorch walk-forward windows through a Hy callback.

```hy
(require lispyn.macros [cross-validate-temporal])
(setv report (cross-validate-temporal time-ordered-dataset score-fold
  :n-splits 4 :test-size 8 :gap 2 :metric "val_loss"))
```

`score-fold(training, validation, fold, repeat)` must build a fresh model and return a finite scalar; `repeat` is always zero.

### `cross-validate-temporal-table`
**Purpose:** Score pandas chronological windows through a Hy callback.

```hy
(require lispyn.macros [cross-validate-temporal-table])
(setv report (cross-validate-temporal-table frame score-fold
  :time-column "timestamp" :n-splits 4 :gap 2
  :metric "val_accuracy" :output "reports/temporal_cv.json"))
```

Keeps training strictly before validation, with no random shuffling.

## Reports, evidence & narratives

### `report-run`
**Purpose:** Show best/final epoch summaries in Jupyter Markdown (or terminal) and optionally save a portable `.md` report.

```hy
(require lispyn.macros [report-run])
(report-run run :title "Baseline results"
  :notes "Synthetic example; evaluate on held-out data."
  :save "reports/baseline.md")
```

Accepts a `Run` from `train-until`/`train` or a mapping of numeric metric histories. Optional `:metric "val_loss"` and `:mode "min"`. Empty unavailable validation series are omitted by default.

### `experiment-check`
**Purpose:** Record a transparent Boolean PASS/FAIL condition with optional observed and expected values.

```hy
(require lispyn.macros [experiment-check])
(setv quality-check
  (experiment-check "Validation accuracy exceeds target"
    (>= val-accuracy 0.8)
    :observed val-accuracy :expected 0.8))
```

Returns a small dict. Options: `:notes`, `:save "reports/check.json"`, `:strict True` (raises on failure). A check must be a real Boolean; `0.8` by itself is not a valid condition.

### `checklist-report`
**Purpose:** Display and optionally save a checklist aggregating `experiment-check` records.

```hy
(require lispyn.macros [checklist-report])
(checklist-report [quality-check]
  :title "Evaluation evidence"
  :save "reports/evidence.json" :strict False)
```

Returns totals and checks; `:strict True` raises if any check fails.

### `narrate`
**Purpose:** Display dynamic Markdown containing measured runtime scalar values.

```hy
(require lispyn.macros [narrate])
(narrate "Validation accuracy was **{accuracy:.1%}**."
  :accuracy val-accuracy :save "reports/finding.md")
```

Placeholders accept simple named fields and restricted numeric format specs. Expressions/attribute access in strings are forbidden; only explicit finite scalar values are inserted. Use `md` for static headings.

### `metric-table`
**Purpose:** Show finite aggregate scalar metrics as a Markdown table in Jupyter/terminal.

```hy
(require lispyn.macros [metric-table])
(metric-table {"val_loss" best-loss "val_accuracy" val-accuracy}
  :title "Validation metrics" :precision 4
  :save "reports/metrics.md")
```

Accepts a nonempty metric-to-real-number mapping (not entire datasets, tensors or prediction arrays). `:precision` is 1–12.

### `experiment-leaderboard`
**Purpose:** Discover saved training histories recursively and display a ranked Markdown leaderboard directly from Hy, without manually collecting run directories.

```hy
(require lispyn.macros [experiment-leaderboard])
(setv board
  (experiment-leaderboard "runs"
    :metric "accuracy" :split "val" :mode "max"
    :rank-by "best" :top 10 :save "runs/leaderboard.md"))
```

Returns a JSON-compatible summary with ranked `runs`, `total` and `skipped` entries. `:save` supports Markdown, CSV and JSON. `:rank-by "final"` ranks each run by the last epoch instead of its best epoch. Nonfinite or missing metrics and unfinished/failed manifests are skipped by default; `:strict True` raises instead. Legacy `history.json` files with no manifest are labeled `history-only`, **not independently verified**. Only scalar histories and completion status are read: no model weights or dataset rows. Compare runs with equivalent splits/protocols before drawing conclusions. See [the runnable Hy demonstration](../examples/experiment_leaderboard.hy).


### `compare-runs`
**Purpose:** Compare a small, **explicit** set of saved training histories with a baseline and display a Markdown table of best/final scores and direction-aware gains in Jupyter or terminal Hy.

```hy
(require lispyn.macros [compare-runs])
(setv comparison (compare-runs
    ["runs/baseline" "runs/candidate"]
    :metric "loss" :split "val" :baseline "baseline"
    :save "reports/compare.md"))
```

Returns a numerical summary, including `best_gain` and `final_gain` (positive means improvement). `:save` accepts `.md`, `.csv` or `.json`. `:mode "min"`/`"max"` overrides metric direction; `:baseline` must match the folder name of one run. Requires 2–20 distinct run directories, each with a valid `history.json`. If a manifest exists it must have `"status": "completed"`. History-only directories remain usable but are not independently certified as completed. **Only compare scores from compatible datasets, seeds and evaluation protocols**—positive gains are descriptive, not a significance test.

### `plot-runs`
**Purpose:** Plot the recorded training or validation metric from multiple **named saved runs** in a single publication-styled figure.

```hy
(require lispyn.macros [plot-runs])
(plot-runs ["runs/baseline" "runs/candidate"]
    :metric "loss" :split "val" :theme "paper"
    :title "Validation loss" :save "figures/comparison.png")
```

Returns a Matplotlib `Figure` and optionally saves `.png` or `.svg`. Supports the `lispyn.plots.plot_comparison` styling arguments (`:palette`, `:figsize`, `:dpi`, etc.). Run histories must contain finite numeric values for the selected metric; run completion is checked when a manifest exists. No model, dataset or training process is loaded or executed. See [the runnable synthetic Hy example](../examples/compare_saved_runs.hy).


---

## Other useful Lispyn commands (not macros)

These are **CLI commands**, not entries from `macros.hy`:

| Command | Purpose |
| --- | --- |
| `lispyn new study.hy` | Create a starter research experiment in Hy |
| `lispyn build study.hy` | Compile to `study.ipynb` without executing |
| `lispyn execute study.hy` | Run and save the Jupyter notebook |
| `lispyn run study.hy` | Execute Hy directly |
| `lispyn parameters study.hy` | Inspect declared `experiment-params` |
| `lispyn build study.hy --set epochs=20` | Override a declared parameter |
| `lispyn sweep study.hy --grid grid.json -o variants/` | Generate bounded, unexecuted parameter variants |
| `lispyn audit study.ipynb --source study.hy` | Audit execution status and verify source hash |
| `lispyn export study.ipynb -o report.md` | Export notebook prose/results/figures to Markdown |
| `lispyn gallery` | Compile supported example notebooks |
| `lispyn compare run-a/ run-b/` | Compare saved experiment histories |
| `lispyn leaderboard runs/ --metric loss -o ranking.md` | Discover and rank saved experiment histories |
| `lispyn report runs/study/` | Summarize a saved run |
| `lispyn doctor` | Inspect environment and optional dependencies |

### Keeping this catalog current

When adding or changing a macro:

1. Implement its public `defmacro` in [`src/lispyn/macros.hy`](../src/lispyn/macros.hy). Backend helpers can live in Python.
2. Add a **`### \`macro-name\``** entry here with the purpose, real Hy usage, arguments/options, return behavior and optional dependencies.
3. Add a working `.hy` example and tests for compilation and direct Hy or Jupyter execution.
4. Run `pytest -q tests/test_macro_catalog.py` (also run automatically in CI). It compares this catalog against the actual macro definitions so newly added macros cannot silently go undocumented.

**Scope:** This index documents exported macros, not every Python helper class or CLI command. [Project overview and installation](../README.md) · [Implementation](../src/lispyn/macros.hy) · [Examples](../examples/).
