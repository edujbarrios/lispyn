# Publishing Lispyn to PyPI

Lispyn uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No long-lived PyPI API token or password is stored in GitHub.

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
   `release: publish v0.20.0a1 [pypi]` (replace the version each time).
   Alternatively create/push an annotated tag `v0.20.0a1` from a tested
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
