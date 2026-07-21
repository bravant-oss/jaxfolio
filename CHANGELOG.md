# Changelog

All notable changes to **jaxfolio** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This file is generated with [git-cliff](https://git-cliff.org) from
[Conventional Commits](https://www.conventionalcommits.org) and curated for clarity.

## [0.1.1] - 2026-07-21

### Performance

- Rewrote the shared constrained solver as a **spectral projected-gradient**
  method (Barzilai–Borwein step sizes), the new default (`solver="spg"`). It
  converges to the same optimum a dedicated QP solver finds — the min-variance
  optimality gap of the old fixed-step Adam solver is gone.
- The built-in optimizers now run through a **jit-cached kernel** keyed on
  problem shape, so repeated solves (e.g. every rebalance of a backtest) compile
  once and reuse the kernel — a single min-variance solve dropped from ~43 ms to
  ~0.2 ms, and walk-forward backtests are dramatically faster.

### Features

- Added an **`examples/benchmark`** suite comparing jaxfolio against
  PyPortfolioOpt, Riskfolio-Lib, skfolio, CVXPY, and SciPy on minimum-variance,
  maximum-Sharpe, and risk-parity — with both a single-solve and an amortized
  rolling-backtest scenario, dark-themed charts, and a fair, shared-moments
  methodology.
- `OptimizerConfig` gained a `solver` option (`"spg"` | `"adam"`) and
  `learning_rate=None` (auto step size); `"adam"` remains available for
  differentiating through the optimizer. Exposed `solve_constrained` and
  `select_projection` from the `toolkit`.

### Documentation

- Added a Benchmark section (chart + results table) to the README and the docs
  home page, and refreshed the solver description in the core-concepts guide.

## [0.1.0] - 2026-07-19

### Features

- Initial public release: sixteen portfolio optimizers (classical, learning, and
  graph-based) behind one `method(returns) → PortfolioResult` interface, a
  differentiable options toolkit, a walk-forward backtester, local-LLM view
  strategies, and dark-themed visualizations — all on a single JAX core.
