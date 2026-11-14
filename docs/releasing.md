# Publishing Lispyn to PyPI

Lispyn uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No long-lived PyPI API token or password is stored in GitHub.

## Release payload: 0.24.0a2

This patch alpha makes the Hy-first narrative starter work for valid training
runs with no separate validation dataset, by omitting empty validation
histories from the automatic `report-run` summary while preserving strict
validation when a specific metric is requested.

The corrected `report-run` behavior is covered by a regression test for
training-only runs, and the full generated-starter notebook is exercised
by the Python 3.11 integration workflow.

Version metadata and package verification are aligned to `0.24.0a2`.
The existing trusted-publishing workflow validates and tags this corrected
version after publishing.

## Release payload: 0.24.0a1

This release advances Lispyn's original Hy-first mission by adding a
`experiment-params` macro that emits Jupyter `parameters`-tagged code
cells, a `report-run` macro for narrative Markdown results and exports,
a documented Hy notebook gallery example and a new experiment scaffold
centered on reproducibility, context and sharing.

To verify the Hy-first notebook story workflow, compile
`lispyn build examples/experiment_story.hy` and inspect the standard
`parameters`-tagged Jupyter code cell and Markdown metric report. The source
uses synthetic histories, not trained model results.

The package, CLI and distribution verification are aligned to `0.24.0a1`.
The official Trusted Publishing workflow validates, builds and publishes
to PyPI before creating the release tag.

## Release payload: 0.23.0a1

This release adds Hy-first chronological train/validation/test splitting,
expanding/rolling walk-forward cross-validation, row-count embargo gaps,
timestamp validation for pandas and native PyTorch adapters. The new temporal
macros support one-call fold-scoring callbacks and privacy-minimal JSON
reports. A runnable synthetic Hy demo is compiled into the notebook gallery.

To verify the Hy-first walk-forward workflow from the checked-out repository,
run `lispyn run examples/temporal_cross_validation.hy` after installing a
native PyTorch build. The example uses synthetic, chronological observations.

Package metadata, CLI version and distribution verification are aligned to
`0.23.0a1`. The official trusted-publishing workflow validates, builds,
uploads to PyPI and tags the successfully published release.

## Release payload: 0.22.0a1

This release adds Hy-first one-call CV macros for PyTorch and pandas,
deterministic repeated stratified/grouped folds, fold-level/weighted summaries,
JSON exports, runnable Hy recipes and regression tests. Native PyTorch handles
training and all previous split APIs remain available.

For a post-install Hy macro smoke test, run the synthetic recipe:
`lispyn run examples/repeated_cross_validation.hy` from the repository root
after installing the package with a matching native PyTorch build.

Package metadata, CLI version and distribution verification are aligned to
`0.22.0a1`. The trusted-publishing workflow validates tests and distributions
before publishing to PyPI and tagging the successful release.

## Release payload: 0.21.0a1

This release adds train/validation/test partitions for pandas and native
PyTorch datasets, plus seeded standard, stratified and group-aware k-fold
cross-validation, out-of-fold score summaries, the `cv-summary` CLI and Hy
macros. Compatibility with the original two-way splits is retained.
The package, CLI, docs and distribution verification are aligned to
`0.21.0a1`. The trusted-publishing workflow runs the release tests,
builds the wheel and source archive, publishes to PyPI and only then
creates the annotated version tag.

## Release payload: 0.20.0a1

This release adds grouped train/validation splits, optional class-stratified
tabular splits and baseline-aware experiment comparisons. Package metadata,
CLI version and distribution verification are aligned to `0.20.0a1`.
The normal trusted-publishing workflow validates the build, tests it and
publishes to PyPI before creating the annotated version tag.

## First alpha: 0.17.0a1 (published)

The initial PyPI Pending Trusted Publisher was registered with:

- Project: `lispyn`
- GitHub repository: `edujbarrios/lispyn`
- Workflow: `release.yml`
- GitHub environment: `pypi`

The first release was successfully published from a dedicated release commit on `main`.
Workflow `.github/workflows/release.yml` validates the exact `pyproject.toml`
version, runs tests, builds both wheel and sdist, checks package metadata
and license files, and uploads the verified artifacts to PyPI using OIDC.
Only **after PyPI confirms success** does the workflow create the matching annotated Git tag.

### GitHub environment

In GitHub, open **Settings → Environments → pypi**. For public releases,
configure required reviewers where available, restrict deployment
branches/tags to your trusted sources, and approve the pending workflow
deployment when it appears. The workflow's publishing job uses the
environment name `pypi` exactly.

### Subsequent releases

1. Update the version in `pyproject.toml`, `src/lispyn/__init__.py`,
   `src/lispyn/cli.py`, and the hardcoded version checks in
   `.github/workflows/package.yml`.
2. Run both the Tests and Package verification workflows on `main`.
3. Create a release commit on `main` with the **exact** subject
   `release: publish v0.24.0a2 [pypi]` (replace the version each time).
   Alternatively create/push an annotated tag `v0.24.0a2` from a tested
   commit. The release workflow verifies the name/version and then publishes
   using the same PyPI Trusted Publisher. For release commits, GitHub
   creates the annotated version tag **after** successful upload.
4. Check [the package on PyPI](https://pypi.org/project/lispyn/), then
   install in a fresh environment:

```bash
python -m pip install "lispyn[notebook,plots]"
lispyn doctor
```

**Do not reuse version numbers:** PyPI distribution files are immutable.
Do not re-tag a failed but partially published release. Bump the version
if a package distribution was already uploaded.

The release workflow is the **only** workflow that can publish to PyPI.
`.github/workflows/package.yml` is a verification-only pipeline.
The distribution does not bundle PyTorch or GPU libraries; users should
install hardware-compatible builds separately. Only execute trusted Hy
programs because compile-time macros and pip-install cells run Python code.
