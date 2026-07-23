# Support

Need help with jaxfolio? Here's where to go.

## Getting help

- **Documentation** — start with the [docs site](https://bravant-oss.github.io/jaxfolio/):
  getting-started guides, per-topic guides, and the full API reference.
- **Questions & discussion** — open a
  [GitHub Discussion](https://github.com/bravant-oss/jaxfolio/discussions) (or an
  issue if Discussions are disabled) for "how do I…" questions, ideas, and
  usage help.
- **Bug reports & feature requests** — open a
  [GitHub Issue](https://github.com/bravant-oss/jaxfolio/issues) using the
  provided templates.
- **Security issues** — follow the private process in [SECURITY.md](SECURITY.md);
  do **not** file them as public issues.

Please search existing issues and discussions before opening a new one.

## Version & compatibility support policy

jaxfolio is pre-1.0 and under active development. This section is the reference
for what we support and how we manage dependencies.

### Python

Supported Python versions: **3.11, 3.12, and 3.13**. These are exercised in CI on
Ubuntu, macOS, and Windows. When a new CPython release becomes stable we add it
to the matrix; we drop a version only after it reaches end-of-life, and such a
change ships in a minor release with a changelog note.

> **Platform note:** JAX support on Windows is
> [community-maintained and less mature](https://docs.jax.dev/en/latest/installation.html)
> than on Linux/macOS. We run the Windows CI leg to catch regressions, but Linux
> and macOS are the primary supported platforms. CPU is the tested target; GPU/TPU
> builds of JAX are supported by JAX itself and are not part of jaxfolio's CI.

### Dependencies

- We specify **minimum** versions for runtime dependencies (`jax`, `jaxlib`,
  `optax`, `numpy`, `pandas`, `scipy`, `matplotlib`) and avoid speculative upper
  bounds, so jaxfolio composes cleanly in larger environments. Upper bounds are
  added only when a specific newer major is known to break us, with a rationale
  recorded inline in `pyproject.toml`.
- For **reproducible** installs, we commit `uv.lock`, which pins exact versions of
  the entire dependency graph. Use `uv sync` (optionally `--frozen`) to reproduce
  a known-good environment. Applications that need determinism should pin via
  their own lockfile too.
- We test against a rolling window of recent releases of the core scientific
  stack. If you hit an incompatibility with a very new or very old transitive
  dependency, please file an issue with your resolved versions
  (`uv pip freeze`).

### API stability

Public API and deprecation guarantees are documented in
[docs/reference/stability.md](docs/reference/stability.md). In short: we follow
SemVer, the public surface is what `jaxfolio.__all__` exports plus documented
submodules, and removals are preceded by a deprecation cycle.

## Commercial / formal support

jaxfolio is provided under the MIT License with no warranty (see
[GOVERNANCE.md](GOVERNANCE.md) for the maintenance commitment). There is no paid
support tier at this time.
