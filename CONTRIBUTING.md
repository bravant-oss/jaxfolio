# Contributing to jaxfolio

Thanks for your interest in improving jaxfolio! This guide covers everything you
need to make a change — from setting up the environment to getting a pull request
merged.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

jaxfolio uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/bravant-oss/jaxfolio
cd jaxfolio
uv sync --all-extras          # core + data + llm + dev + docs
```

That installs the package in editable mode along with every optional extra, so
the full test suite and docs build work out of the box.

## Everyday commands

```bash
uv run pytest                              # run the test suite
uv run pytest --cov=jaxfolio               # with coverage (must stay above the floor)
uv run ruff check .                        # lint
uv run ruff format .                       # auto-format
uv run ruff format --check .               # verify formatting (what CI runs)
uv run mkdocs serve                        # preview docs locally
uv run mkdocs build --strict               # build docs the way CI does
```

### Pre-commit

We ship a [pre-commit](https://pre-commit.com/) config that runs Ruff and a few
hygiene hooks before each commit:

```bash
uv run pre-commit install       # one-time, installs the git hook
uv run pre-commit run --all-files
```

## Coding conventions

- **Style & linting** are enforced by Ruff (`line-length = 100`, rule set in
  `pyproject.toml`). Run `ruff format` before committing.
- **Docstrings use the NumPy style** (`Parameters` / `Returns` sections). The
  docs site renders public APIs from these via mkdocstrings, so every public
  function/class needs at least a one-line summary, and non-trivial ones need
  full `Parameters`/`Returns`. Cite papers where relevant (existing code does).
- **Public vs. private surface.** The public API is the set of names exported in
  `src/jaxfolio/__init__.py`'s `__all__` plus the documented submodule entry
  points. Anything underscore-prefixed (e.g. helpers in `optimizers/base.py`) is
  internal and may change without notice. See
  [API stability](docs/reference/stability.md).
- **JAX first.** New numerical code should be JAX-native and differentiable where
  it makes sense, reusing the shared building blocks in `jaxfolio.toolkit`
  (`moments`, `make_projection`, `solve_projected_gradient`, `finalize_result`)
  rather than re-implementing solvers.

## Tests

- Every bug fix and feature needs a test. Put unit tests under `tests/`; put
  reference-solver / numerical-validation tests under `tests/validation/`.
- Reuse the shared fixtures in `tests/conftest.py` (`returns`, `small_returns`).
- **Coverage floor.** CI enforces a minimum total coverage via `fail_under` in
  `pyproject.toml` (`[tool.coverage.report]`). We *ratchet it up over time*: when
  a change meaningfully raises the measured coverage, bump `fail_under` toward it
  in the same PR. Never lower the floor to make a PR pass — add tests instead.

## Commits & changelog

This project follows [Conventional Commits](https://www.conventionalcommits.org)
(`feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `refactor:`, `test:`). The
`CHANGELOG.md` is generated from commit history with
[git-cliff](https://git-cliff.org) (`cliff.toml`) and curated for clarity, so a
clean commit history directly becomes good release notes.

When your change deprecates or removes public API, follow the
[deprecation policy](docs/reference/stability.md) and add a note under a
`Deprecations` heading in the changelog.

## Pull requests

1. Branch from `main` (e.g. `feat/vol-surface`, `fix/hrp-nan`).
2. Keep PRs focused; open an issue first for large or cross-cutting changes.
3. Ensure `ruff check`, `ruff format --check`, `pytest` (with coverage), and
   `mkdocs build --strict` all pass locally — CI runs the same on Ubuntu, macOS,
   and Windows across Python 3.11–3.13.
4. Fill out the PR template and link the issue it closes.

## Questions

See [SUPPORT.md](SUPPORT.md) for where to ask questions and how to report bugs.
Security issues follow a private process — see [SECURITY.md](SECURITY.md).
