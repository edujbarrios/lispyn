# Supported Lispyn macros

**The complete Hy/Lisp macro reference — 49 macros, with usage examples.**

Lispyn's API is **Hy macros**. They compose ordinary Python, pandas, PyTorch, Matplotlib and Jupyter; they do not replace those libraries. This file is the **single source of truth for supported macros**, grouped by research workflow. The implementations live in [`src/lispyn/macros.hy`](../src/lispyn/macros.hy).

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

| Research activity | Macros |
| --- | --- |
| [Notebook structure & parameters](#notebook-structure--parameters) | `md`, `markdown`, `raw-cell`, `cell-tags`, `pip-install`, `experiment-params` |
| [NumPy and pandas data profiling](#numpy-and-pandas-data-profiling) | `array-profile`, `table-profile` |
| [PyTorch training & inference](#pytorch-training--inference) | `with-seed`, `train-for`, `train-until`, `train-accumulating`, `with-eval`, `head-tune`, `full-tune` |
| [Runtime dependencies](#runtime-dependencies) | `ensure-package`, `use-package` |
| [Visualization & classification](#visualization--classification) | `curve`, `defcurve`, `compare-curves`, `learning-rate-curve`, `confusion-matrix`, `defconfusion`, `classification-report` |
| [Dataset splits & cross-validation](#dataset-splits--cross-validation) | `train-val-test`, `cross-validation-folds`, `tabular-train-val-test`, `tabular-cross-validation-folds`, `repeated-cross-validation-folds`, `repeated-tabular-folds`, `cross-validate`, `cross-validate-table`, `save-cross-validation` |
| [Time-series experiments](#time-series-experiments) | `temporal-train-val-test`, `temporal-folds`, `temporal-table-split`, `temporal-table-folds`, `cross-validate-temporal`, `cross-validate-temporal-table` |
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
**Purpose:** Temporarily use PyTorch evaluation and inference mode; restore the model's previous training state even on error.

```hy
(require lispyn.macros [with-eval])
(setv logits (with-eval model (model features)))
```

Arguments: `model` plus Hy body. Wraps `torch.inference_mode()`, calls `model.eval()` and restores training mode.

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
