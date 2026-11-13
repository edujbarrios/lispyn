# Publishing Lispyn to PyPI

Lispyn uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No long-lived PyPI API token or password is stored in GitHub.

## First alpha: 0.17.0a1

The initial PyPI Pending Trusted Publisher was registered with:

- Project: `lispyn`
- GitHub repository: `edujbarrios/lispyn`
- Workflow: `release.yml`
- GitHub environment: `pypi`

The first release is triggered by the dedicated release commit on `main`.
Workflow `.github/workflows/release.yml` validates the exact `pyproject.toml`
version, runs tests, builds both wheel and sdist, checks package metadata
and license files, and uploads the verified artifacts to PyPI using OIDC.
Only **after PyPI confirms success** does the workflow create the
`v0.17.0a1` Git tag.

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
3. Create/push an annotated Git tag with the matching version (for example
   `git tag -a v0.18.0a1 -m 'Lispyn 0.18.0a1'` and
   `git push origin v0.18.0a1`). The release workflow checks that the
   tag and package version match and publishes automatically after
   any required environment approval.
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
