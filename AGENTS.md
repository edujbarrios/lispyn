
# Working with Lispyn — humans and AI coding agents

Lispyn is a Hy macro library for authoring native Python/PyTorch AI experiments
and reproducible Jupyter notebooks. It is **not** an autonomous training system,
a sandbox, or a replacement for the libraries underneath.

## Safe authoring workflow

1. Start with `examples/hello.hy`, `examples/showcase.hy` or
   `examples/macro_composition.hy`; look up macro signatures in
   `supported_macros/README.md`.
2. Add `(require lispyn.macros [...])` to **every** independent code cell that
   uses macros. Prefer ordinary PyTorch/Python when macros add no real structure.
3. Inspect generated Python with `hy2py path/to/study.hy`; compile the notebook
   with `lispyn build path/to/study.hy` before execution.
4. Run `pytest -q` and relevant CPU examples before changing macros.
   Test success, exceptions, expression evaluation counts and restored states.
5. Human reviewers should approve source, data access, dependency installs,
   execution and publishing when an AI agent contributes.

## Security and reproducibility

**Building runs Hy macro expansion (Python code) at compile time.**
The build step is not a sandbox: never compile untrusted Hy sources.
`pip-install` creates a dependency cell that executes only when run, but its
requirements still need review. No macro guarantees scientific correctness,
cross-device reproducibility, absence of leakage or model quality.
Use held-out evaluation and record experiment provenance.

## Development boundaries

- Public source forms belong in `src/lispyn/macros.hy`, with matching
  catalog entries headed by the macro name and real executable Hy examples.
- Use `hy.gensym` for generated internal variables, evaluate effectful
  expressions once, and use `try/finally` to clean up state.
- Prefer visible native PyTorch, Matplotlib and Python control flow over new
  runtime backends. PyTorch remains optional at package-install time.
- Follow the versioning and release policy in `.github/workflows/release.yml`.
  Never tag a version before tests and publication succeed.
