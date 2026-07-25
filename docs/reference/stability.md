# API stability & deprecation policy

jaxfolio follows [Semantic Versioning](https://semver.org). This page defines
what "the API" means, what you can rely on across releases, and how we retire
things when they must change.

## Pre-1.0 status

jaxfolio is currently in the `0.y.z` series (`Development Status :: 4 - Beta`).
Under SemVer, while the major version is `0` the API is still stabilizing:
breaking changes may occur in a **minor** (`0.y`) release. We nonetheless apply
the deprecation process below wherever practical, so upgrades stay smooth, and we
document every breaking change in the [changelog](https://github.com/bravant-oss/jaxfolio/blob/main/CHANGELOG.md).

Once we reach `1.0.0`, the standard SemVer contract applies: breaking changes to
the public API only in a new **major** version.

## What is public

The **public API** — the surface covered by these guarantees — is:

- Every name exported from the top-level package, i.e. the entries in
  `jaxfolio.__all__` (`import jaxfolio as jf; jf.maximum_sharpe`, etc.).
- The public names of the documented submodules that back the guides and the
  [API reference](index.md): `jaxfolio.optimizers`, `jaxfolio.options`,
  `jaxfolio.llm`, `jaxfolio.backtest`, `jaxfolio.data`, `jaxfolio.moments`,
  `jaxfolio.constraints`, `jaxfolio.viz`, `jaxfolio.toolkit`, `jaxfolio.types`,
  and `jaxfolio.registry`.

Everything else is **internal** and may change or disappear without a deprecation
cycle, including:

- Any name prefixed with an underscore (e.g. `_solve_cached`, helpers in
  `jaxfolio.optimizers.base`, `jaxfolio._deprecation`).
- Undocumented modules, attributes, and function internals.
- Exact numerical values that are not part of a documented contract (e.g. the
  precise iterate path of the SPG solver), as opposed to documented properties
  (weights sum to one, min-variance optimality, etc.).

If you depend on something internal, please open an issue so we can consider
promoting it to the public API.

## Experimental features

Some public features are explicitly **experimental** and are exempt from the
stability guarantees until promoted. They are marked in their docstrings and emit
a warning at runtime:

- **LLM strategies** (`llm_black_litterman`, `llm_sentiment_portfolio`,
  `llm_agent_portfolio`) — emit an `ExperimentalWarning` on first use. Their API
  and behavior may change in any release. See [DISCLAIMER.md](https://github.com/bravant-oss/jaxfolio/blob/main/DISCLAIMER.md).

## Deprecation process

When a public name must change or be removed:

1. **Deprecate, don't delete.** The old name keeps working and emits a
   `DeprecationWarning` (via `jaxfolio._deprecation.deprecated` /
   `warn_deprecated`) that names the removal version and the replacement.
2. **Grace period.** The deprecated name remains for **at least one minor
   release** before removal.
3. **Document it.** Every deprecation is recorded under a `Deprecations` heading
   in `CHANGELOG.md`, and the removal is noted when it happens.

### Seeing deprecation warnings

`DeprecationWarning` is silenced by Python by default. To surface jaxfolio's
warnings in your own code:

```bash
python -W "default::DeprecationWarning" your_script.py
```

jaxfolio's own test suite turns `DeprecationWarning`s from the `jaxfolio` package
into errors, so internal callers can never silently rely on a deprecated path.

## Dependencies & platforms

Supported Python versions, the dependency-versioning policy, and platform support
are documented in [SUPPORT.md](https://github.com/bravant-oss/jaxfolio/blob/main/SUPPORT.md).
