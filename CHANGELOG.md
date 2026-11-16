# Changelog

## 0.34.0a1 — 2026-11-16 — Realización de notebooks de IA and class balance

- Reposition Lispyn as a general AI notebook authoring and execution tool, not exclusively a research experiment framework; update README, package metadata, CLI description and citations.
- Add `label-balance`: inspect class counts, proportions and imbalance ratios for sequences, NumPy arrays, pandas Series and optional PyTorch tensors, with Markdown/JSON exports.
- Add `plot-label-balance`: create Matplotlib class-count figures with existing Lispyn themes and optional PNG/SVG exports.
- Include a runnable `examples/label_balance.hy` covering source execution, notebook compilation and notebook execution, plus edge-case and documentation tests.
- Preserve the three modes (`run`, `build`, `execute`), dynamic release/CI badges, and actual source-backed README examples. Document all 53 macros in the full catalog.


## 0.33.0a1 — 2026-11-16 — Real README evidence and input validation

- Add `assert-finite` for NumPy numerical arrays and optional PyTorch dense tensors, rejecting NaN/inf before training without printing raw values.
- Add `assert-shape` for NumPy arrays, pandas DataFrames and PyTorch tensors, including wildcard batch dimensions.
- Validate the new macros with Python unit tests and a runnable Hy example covering `run`, `build` and `execute`.
- Back every research benefit in the root README with code excerpts taken verbatim from `examples/quickstart.hy` and `examples/validate_inputs.hy`; add regression tests preventing drift.
- Preserve the three clear execution commands, dynamic badges, and a prominent link to the complete 51-macro catalog.


## 0.32.0a1 — 2026-11-16 — Data profiling and concise research onboarding

- Add `array-profile` for NumPy shape, dtype, NaN/infinity counts and finite-only summary statistics.
- Add `table-profile` for pandas missingness, numeric infinities, duplicate rows and per-column dtypes.
- Support terminal Markdown, Jupyter Markdown and aggregate .md/.json exports without including dataset rows or array elements.
- Add a complete runnable Hy example and regression coverage for direct execution, unexecuted notebook builds and executed notebook output.
- Make the root README concise: purpose, research benefits, installation and all three execution commands; send macro usage to the 49-entry reference.
- Keep verifying genuine executed PyTorch notebooks in CI without inserting large rendered examples into the landing README.


## 0.31.0a2 — 2026-11-16 — Clearer research onboarding and reliable release checks

- Rewrite the README for researchers: what Lispyn offers, when to use it, and three separately runnable commands against one small `hello.hy` script.
- Explain the difference between running Hy directly, building an unexecuted Jupyter notebook, and executing/saving notebook outputs.
- Fix the previous `assert-close` recursion and ensure `with-timing` expands in independent notebook cells.
- Make package CI assert the declared version instead of hard-coding `0.30.0a1`, and cover all three hello workflows with regression tests.
- Retain the real executed PyTorch quickstart and provenance-backed notebook example. No new training dependencies.


## 0.31.0a1 — 2026-11-16 — Hello Lispyn and lightweight diagnostics

- Add `assert-close` for finite scalar checks with configurable numeric tolerances, with clear failures for invalid or unequal values.
- Add `with-timing` to record labeled wall-clock step duration in ordinary Hy execution and Jupyter, including exceptional exits.
- Introduce `examples/hello.hy`, a tiny PyTorch-free script using both new macros and an authored Markdown cell.
- Document **all three** execution workflows with the **same hello.hy**: `lispyn run` (direct Hy), `lispyn build` (compile notebook only), and `lispyn execute` (compile and execute notebook).
- Expand the supported macro reference to 47 macros and add smoke and validation tests for helpers and the three CLI pathways.
- Keep the native PyTorch research workflows and existing public API compatible; the new macros require no additional third-party packages.


## 0.30.0a1 — 2026-11-16 — Compare experiments without leaving Hy

- New Hy `compare-runs` macro: compare 2–20 explicitly selected saved PyTorch run directories, report best/final epoch scores and direction-aware improvements against a named baseline directly as rich Jupyter Markdown (or terminal output).
- New Hy `plot-runs` macro: draw named recorded metric curves as Matplotlib figures using Lispyn themes and figure exports, with no retraining and no ad hoc Python.
- Both macros work with `history.json` records, reject invalid/ambiguous paths and invalid completion manifests; the existing Python comparison/plotting backends remain available.
- Add a runnable **synthetic** Hy example, gallery notebook and tests covering baseline gains, metric direction, validation, exports, real Hy execution and live Jupyter Markdown/image output.
- Expand `supported_macros/README.md` to **45/45 documented macros**, preserving the concise README, publication-ready notebooks and native PyTorch interoperability.

Comparison gains are descriptive evidence, not statistical significance tests: use consistent datasets, evaluation protocols and seeds before comparing runs.


## 0.29.0a1 — 2026-11-16 — Cite, package and verify experiments

- Add root `CITATION.cff` (Citation File Format 1.2.0) so GitHub can offer **Cite this repository** for the original software author and version.
- Add `lispyn capsule SOURCE.hy NOTEBOOK.ipynb -o capsule.zip [--run RUN_DIR]`: package a verified, fully executed notebook with the original Hy source into a deterministic, portable ZIP with a SHA-256 file manifest. Optional completed runs contribute history and manifest JSON only, not model weights or datasets.
- Add `lispyn verify-capsule capsule.zip [--json]`: recheck all member hashes, provenance, notebook execution state and safe archive structure offline, without extracting or executing anything. SHA-256 integrity is not a cryptographic signature and does not prove scientific validity.
- Reduce the README to a concise, researcher-facing overview while retaining a real Hy code excerpt, **recorded** notebook outputs and the actual training chart. Keep full reference material in `supported_macros/README.md`.
- Add regression coverage for reproducible ZIPs, successful and rejected capsules, tampering, unsafe paths, CLI output, CFF metadata and README result provenance.

The integrity manifest is designed to catch accidental changes, not to certify author identity. Capsules contain notebook output, so review each cell before sharing outside your research group.


## 0.28.0a2 — 2026-11-16 — One-command notebook and plotting installation

- Make `pip install lispyn` install Jupyter notebook execution (`nbclient`, `ipykernel`) and Matplotlib (`matplotlib`) without extra selectors. Keep `[notebook]` and `[plots]` extras compatible with existing installs.
- Introduce an optional root `requirements.txt` for a cloned research environment with the Lispyn source, PyTorch, NumPy, pandas, Jupyter and Matplotlib. PyTorch deliberately stays outside the base package because suitable CPU/CUDA/ROCm wheels vary by platform.
- Simplify the main README onboarding and document the `pip-install` Hy macro for an explicitly requested optional dependency cell. The macro installs only when an experiment notebook is executed; compiling a notebook never installs anything.
- Verify default wheel metadata, isolated wheel imports, research requirements, and the nonexecuting notebook dependency macro with new tests; keep the existing 43-macro public API and PyTorch/Jupyter workflow unchanged.

This release is intentionally **not** a one-size-fits-all PyTorch installer: use the official platform-matched build for accelerators and `requirements.txt` only for an optional full research environment from a clone.


## 0.28.0a1 — 2026-11-16 — A new way to write and compare experiments

- Present Lispyn as **a new way to write and run machine-learning experiments**, with a concise researcher-first README and PyPI description.
- Add `experiment-leaderboard` Hy macro: automatically discover saved run histories and render ranked, shareable Markdown tables in Jupyter or terminal Hy.
- Add `lispyn leaderboard DIR [--metric loss] [--rank-by best|final] [--mode min|max] [--top N] [--strict] [--json] [-o report.md|csv|json]` for offline experiment ranking.
- Default to completed native runs; explicitly mark history-only artifacts. Skip missing/incomplete/nonfinite results with diagnostics or use strict mode. Read only aggregate histories and optional manifest completion status, never datasets or model checkpoints.
- Add a Hy example, notebook gallery entry, full macro catalog entry and end-to-end tests for ranking, deterministic ties, output formats, failures and Jupyter Markdown.
- Keep existing PyTorch native training, notebook generation and source provenance APIs compatible.

This pre-release keeps ranking reports descriptive: scientific comparisons still require consistent evaluation datasets, protocols and seeds.


## 0.27.0a1 — 2026-11-16 — Dynamic Hy experiment stories and notebook audit

- Add `narrate` Hy macro to interpolate finite, explicit runtime scalar values into Jupyter Markdown with restricted named placeholders and numeric formatting, and optionally save the result to a Markdown file.
- Add `metric-table` Hy macro to display and export compact Markdown metric summaries from explicit finite numeric values.
- Add `lispyn audit NOTEBOOK.ipynb [--source ORIGINAL.hy] [--json]`: nonexecuting readiness check for executed code cells, stored error outputs, Hy source hashes and verified provenance; return failure status for incomplete or mismatched artifacts.
- Include the synthetic `examples/living_narrative.hy` in the notebook gallery, plus tests for direct Hy compilation/execution, live Jupyter evaluation, safe narrative templates, and offline provenance auditing.
- Retain existing PyTorch, parameterization, evidence, Markdown export and notebook workflows without breaking changes.


## 0.26.0a1 — 2026-11-14 — Evidence-backed Hy experiments and notebook publishing

- Add `experiment-check` and `checklist-report` Hy macros for explicit Boolean success criteria, visible PASS/FAIL notebook narratives, optional strict failures, and compact JSON evidence summaries.
- Add `lispyn export NOTEBOOK.ipynb -o REPORT.md [--hide-code]` for offline, nonexecuting conversion of Jupyter Markdown, safe text outputs, and PNG/JPEG figures into a portable Markdown report plus assets.
- Preserve Hy source-name/SHA-256 provenance in exported reports; omit notebook HTML/JavaScript/SVG outputs and refuse to overwrite prior exports or image assets.
- Add a synthetic Hy evidence-story example to the generated notebook gallery and tests for executable macros, report aggregation, HTML output omission, image extraction and export safety.
- Keep existing PyTorch training, cross-validation, parameters/sweep and notebook APIs unchanged.


## 0.25.0a1 — 2026-11-14 — Reusable and parameterized Hy experiments

- Turn existing `experiment-params` macros into reusable notebook configurations through `lispyn build/execute --set NAME=JSON` and `--params FILE`.
- Add `lispyn parameters SOURCE --json` to inspect literal defaults and nonliteral expressions without executing notebook code.
- Add `lispyn sweep SOURCE --grid FILE -o DIR` to build deterministic, bounded (64 max), unexecuted Jupyter variants with a JSON provenance index.
- Apply validated JSON literals to the compiled Jupyter `parameters` cells without editing the Hy source, evaluating user expressions, or accepting undeclared names.
- Stamp each notebook with a SHA-256 of its originating Hy source; keep cell identities reproducible across identical parameterized builds.
- Add Hy example and gallery coverage plus end-to-end tests for execution, variants, validation and safety; preserve the old build/run flows.


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
