# Changelog

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
