# Releasing Lispyn to PyPI

Lispyn's source uses editable installs for development. The public PyPI package
will support regular installs once a release is approved and actually uploaded.
**This repository does not automatically publish to PyPI.**

## Before the first release

1. Verify that the distribution name \`lispyn\` is available and usable by this
   project's owner on [PyPI](https://pypi.org/project/lispyn/). A search result
   alone does not reserve the package name.
2. Ensure the author and license notices are correct, and choose a version
   not already published. PyPI artifacts are immutable: do not reuse versions.
3. Run the test workflow and the packaging workflow on a clean commit.
4. Inspect the wheel and source archive in GitHub Actions. They must include
   \`lispyn/macros.hy\`, \`LICENSE\`, \`NOTICE\`, the console entry point and
   the correct version metadata.
5. Optionally try **TestPyPI** with a separate account and Trusted Publisher.
   TestPyPI dependency resolution may need the main PyPI index.

## Local release preparation

\`\`\`bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
\`\`\`

For developing Lispyn use:

\`\`\`bash
python -m pip install -e ".[dev,notebook,plots]"
lispyn doctor
\`\`\`

After an actual PyPI release, an end user should be able to run:

\`\`\`bash
python -m pip install "lispyn[notebook,plots]"
lispyn doctor
\`\`\`

The \`package.yml\` GitHub workflow performs a wheel+sdist build,
\`twine check\`, distribution-content validation, and a clean virtual
environment install/Hy macro smoke test. It **uploads build artifacts to
GitHub Actions only**. It cannot upload packages to PyPI.

## Publishing with Trusted Publishing

Use [PyPI's Trusted Publishing instructions](https://docs.pypi.org/publishing/publishing-to-pypi/)
rather than a permanent personal API token. Register the GitHub repository
\`edujbarrios/lispyn\` and a specific publication workflow on PyPI, use a
protected \`pypi\` GitHub environment with **manual approval**, and give that
job \`id-token: write\` permission.

Create or enable an actual upload workflow only when the maintainer approves
a release and the package name/account ownership is confirmed. This prevents
unintended publishing on arbitrary pushes. Include an explicit Git tag or
manual dispatch gate, and do not publish from the test workflow itself.

## Compatibility notes

Lispyn is pure Python/Hy; the wheel does **not bundle** PyTorch or compiled
audio/video libraries. Users should select compatible PyTorch, TorchVision,
TorchAudio, TorchCodec and FFmpeg builds for their hardware. The optional
\`peft\`, \`vision\`, \`audio\` and \`plots\` extras are separate.

Only run trusted Hy sources: Hy macros may execute during compilation, and
a runtime \`pip-install\` cell can install third-party code.
