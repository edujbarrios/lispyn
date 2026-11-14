# Changelog

## 0.24.0a2 — 2026-11-14 — Correct reports for training without validation

- Fix `report-run` to omit empty, unavailable validation histories from multi-metric summaries, such as training without a validation loader.
- Keep explicit `:metric "val_loss"` strict: selecting an empty metric still reports a clear error.
- Add regression coverage for the real `lispyn new` baseline workflow.
- Supersedes 0.24.0a1 for experiments generated with the new narrative starter.


## 0.24.0a1 — 2026-11-14 — Hy experiment narratives and shareable reports

- Add `experiment-params` macro: write ordinary `setv` declarations once in Lisp and compile them into a standard Jupyter `parameters`-tagged cell for reproducible notebook parameterization.
- Add `report-run` macro: produce readable per-metric best/final epoch summaries, Markdown display in Jupyter or terminal output, and optional portable `.md` export.
- Include a narrative-focused Hy example in the Jupyter gallery and update the starter from `lispyn new` to include parameters, hypotheses, training and results.
- Extend regression coverage for compiled parameter cells, direct Hy execution, stable notebook IDs, report formatting, invalid inputs, and exports.
- Keep previous Hy macros, PyTorch APIs and notebook formats backward compatible; no training data is included in exported metric summaries.


## 0.23.0a1 — 2026-11-13 — Leakage-aware chronological experiments

- Add Hy-first temporal train/validation/test macros and chronological expanding/rolling walk-forward CV for native PyTorch datasets and pandas.
- Add configurable row-count embargo gaps, minimum/maximum training window lengths, and explicit validation-window sizes.
- Validate optional table timestamps (sorted, unique, nonmissing), avoid mutating input frames and reject impossible temporal windows.
- Add one-call `cross-validate-temporal` and `cross-validate-temporal-table` macros, reusing privacy-minimal fold reports and JSON export.
- Add a runnable synthetic Hy/PyTorch temporal training example to the Jupyter notebook gallery.
- Add temporal leak-prevention tests, invalid-argument tests and Hy macro compilation/execution tests; retain previous cross-validation APIs.


## 0.22.0a1 — 2026-11-13 — Hy-first repeated cross-validation

- Add `cross-validate` and `cross-validate-table` Hy macros to run a fold-scoring callback directly from Lisp, with explicit fresh model initialization per fold.
- Add reproducible repeated k-fold CV for pandas and PyTorch, supporting class stratification or group-leakage prevention without scikit-learn.
- Report per-fold scores and sizes, mean, standard deviation, min/max and validation-size weighted mean, with a privacy-minimal JSON export macro.
- Add `repeated-cross-validation-folds`, `repeated-tabular-folds`, `tabular-train-val-test` and `tabular-cross-validation-folds` Hy macros.
- Include working synthetic PyTorch Hy demonstration in the generated Jupyter gallery; extend tests for Hy execution, deterministic folds and grouped isolation.
- Preserve existing CV APIs, macros and the original two-way/three-way split defaults.


## 0.21.0a1 — 2026-11-13 — Held-out test splits and k-fold cross-validation

- Add reproducible train/validation/test splits for pandas and PyTorch, with optional stratification or leakage-safe group boundaries.
- Add 2–N k-fold cross-validation on either tabular frames or indexable PyTorch datasets, including class-balanced and group-aware folds.
- Add pure-Python cross-validation summaries (mean, population standard deviation, min/max) and a `lispyn cv-summary` JSON CLI exporter.
- Add `train-val-test` and `cross-validation-folds` Hy macros and a synthetic, offline example.
- Add regression coverage for disjoint partitions, exactly-once out-of-fold coverage, reproducible seeds and invalid configurations.
- Keep existing two-way split APIs and defaults unchanged.


## 0.20.0a1 — 2026-11-13 — Leakage-safe data splits and baseline comparisons

- Add group-aware train/validation splits for DataFrames and arbitrary PyTorch datasets, keeping all records for a subject or session in one partition.
- Add optional seeded, per-class stratification to tabular DataFrame splits without requiring scikit-learn.
- Add `lispyn compare --baseline RUN` with direction-aware best/final gains, included in Markdown, CSV and JSON exports.
- Add regression coverage for deterministic splits, non-overlapping groups, invalid inputs, baseline reporting and CLI output.
- Retain legacy random split and comparison formats when the new options are not requested.


## 0.19.0a1 — 2026-11-13 — Advanced native PyTorch training controls

- Add true gradient accumulation with correctly normalized incomplete final windows.
- Add optional global gradient-norm clipping before optimizer updates.
- Add cosine, step and plateau learning-rate schedulers, plus custom native PyTorch factories.
- Record the learning rate at each epoch and add a themed learning-rate plot macro.
- Add real CPU training and scheduler tests and a generated Hy/Jupyter demonstration.
- Keep the existing defaults unchanged for users who do not enable these features.

## 0.18.0a1 — 2026-11-13 — Richer notebooks, experiment comparisons and reports

- Fix README package badges: official PyPI version/Python support, main-branch CI, explicit Apache-2.0 license.
- Add `classification-report` Hy macro and precision/recall/F1/support calculations with Markdown/JSON export, no scikit-learn dependency.
- Add `raw-cell` and `cell-tags` macros for first-class Jupyter raw cells and cell metadata.
- Add `lispyn compare` CLI with cross-run loss/accuracy summaries, Markdown/CSV/JSON export and optional combined SVG/PNG plots.
- Extend generated gallery and synthetic classification report example.
- Improve reusable PyPI workflow to validate each release version and tag after successful trusted publishing.

## 0.17.0a1 — First public PyPI alpha

The first Lispyn alpha packages the Hy-to-Python Jupyter workflow.

### Highlights

- Compile `.hy` sources into ordinary Python Jupyter notebooks with authored Markdown cells.
- Declare notebook dependency cells with the `(pip-install ...)` compile-time macro; installed requirements are detected at runtime.
- Use native PyTorch, TorchVision, TorchAudio and optional PEFT libraries directly from Hy.
- Reuse training macros, including `head-tune`, `full-tune`, `train-until`, and deterministic seeding.
- Track experiment histories, best/last checkpoints, early stopping, and a manifest without storing dataset samples.
- Plot training histories and multi-run comparisons with reusable Matplotlib themes, including `docusaurus-dark` and `turquoise-spectrum`.
- Generate confusion matrices and a checked-in gallery of unexecuted example notebooks.
- Inspect local dependencies with `lispyn doctor`.

### Installation

```bash
python -m pip install "lispyn[notebook,plots]"
lispyn doctor
```

The initial release is **experimental** and has no stability guarantees. The
package uses Hy on Python, not a separate notebook kernel, and does not bundle
PyTorch, GPU runtimes or pretrained model weights.

Original creator: **Eduardo J. Barrios**. Licensed under Apache-2.0.
