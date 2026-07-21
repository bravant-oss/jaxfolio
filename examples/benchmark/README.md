# Benchmark: jaxfolio vs. popular portfolio-optimization libraries

A head-to-head comparison of **jaxfolio** against the best-known Python
portfolio-optimization libraries on three canonical long-only, fully-invested
problems:

| Problem | Objective |
|---|---|
| **Minimum Variance** | minimize `wᵀΣw` |
| **Maximum Sharpe** | maximize `(wᵀμ − r_f) / √(wᵀΣw)` (tangency portfolio) |
| **Risk Parity (ERC)** | equalize each asset's contribution to total risk |

Every library is fed the **same moments** (`μ`, `Σ` — per-period sample
estimates matching jaxfolio's `sample_covariance`) wherever its API allows, so
the comparison is apples-to-apples rather than a comparison of moment
estimators. skfolio estimates its own moments with its default empirical prior
(which matches these); that shows up as a small weight disagreement, not an
unfair edge.

## Libraries compared

| Library | Install | Min-Var | Max-Sharpe | Risk-Parity |
|---|---|:-:|:-:|:-:|
| **jaxfolio** (reference) | — | ✅ | ✅ | ✅ |
| [SciPy](https://scipy.org/) SLSQP baseline | *(bundled)* | ✅ | ✅ | ✅ |
| [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | `pip install PyPortfolioOpt` | ✅ | ✅ | — |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | `pip install Riskfolio-Lib` | ✅ | ✅ | ✅ |
| [skfolio](https://skfolio.org/) | `pip install skfolio` | ✅ | ✅ | ✅ |
| [CVXPY](https://www.cvxpy.org/) | `pip install cvxpy` | ✅ | ✅ | ✅ |

SciPy is always available (jaxfolio already depends on it) and serves as a
neutral SLSQP reference. The other four are **optional**: the benchmark
auto-detects which are installed and silently skips the rest, so it runs out of
the box and gets richer as you install more competitors. PyPortfolioOpt ships
HRP rather than equal-risk-contribution risk parity, so it is not compared on
that problem (comparing different methods would be misleading).

## Running

```bash
# Runs immediately with jaxfolio + SciPy:
uv run python examples/benchmark/benchmark.py

# Add the famous competitors for the full picture:
uv pip install PyPortfolioOpt Riskfolio-Lib skfolio cvxpy
uv run python examples/benchmark/benchmark.py
```

## What it measures

For every (problem × library) pair the script reports:

- **Speed** — median wall-clock solve time over several runs. jaxfolio gets a
  warm-up run first, so JAX tracing/compilation is *not* charged to the timed
  runs (this is the fair way to time a JIT library).
- **Quality** — the achieved objective, recomputed by the benchmark from the one
  shared moment set, so the number means the same thing for every library
  (annualized volatility, annualized Sharpe, or the coefficient of variation of
  risk contributions where `0` = perfect parity).
- **Agreement** — `Δw`, the largest per-asset weight difference from jaxfolio.
  Near-zero means the libraries converge to the same portfolio.

It runs **two scenarios**:

1. **Single solve (cold)** — one solve on a 20-asset panel. jaxfolio pays its
   one-time JIT compilation here (a warm-up run absorbs it before timing).
2. **Repeated solves** — every library solves 60 rolling look-back windows, the
   workload of a monthly-rebalanced backtest. jaxfolio compiles once and reuses
   the kernel across all 60 windows (same shape ⇒ jit-cache hit); the reported
   `ms/solve` is amortized. This is the regime jaxfolio is built for.

Output (written to `examples/benchmark/output/`):

- `benchmark_results.csv` / `benchmark_throughput.csv` — the tidy result tables.
- `benchmark_speed.png` — single cold-solve time per problem (log scale).
- `benchmark_throughput.png` — amortized ms/solve over the rolling backtest.
- `benchmark_weights.png` — min-variance weights across libraries, side by side.

## Reading the results

A few things worth knowing when you interpret the numbers:

- **jaxfolio's constrained solvers are one projected-gradient routine.** Min-var,
  mean-var, max-Sharpe, max-diversification and Kelly are all the *same*
  jit-compiled solver with a different objective, chosen so the whole pipeline
  stays end-to-end differentiable (you can take a gradient *through* an
  allocation). The default is a spectral projected-gradient method
  (Barzilai-Borwein step sizes), which converges to the **same optimum a
  dedicated QP solver finds** — the quality columns match the other libraries to
  ~4 decimals. Risk parity uses exact cyclical coordinate descent (perfect
  equal-risk-contribution); CVaR keeps the Adam solver.
- **jaxfolio is fastest where it's reused.** Because each built-in optimizer is a
  cached JIT kernel keyed on problem shape, a backtest that re-solves the same
  20-asset problem 60 times compiles *once* — so the amortized `ms/solve` is well
  below the QP libraries (often 5–80×). On a single cold solve jaxfolio is still
  competitive-to-fastest, but that is not the workload it's optimized for;
  differentiating through the optimizer to train an allocation policy is the
  other (and a single-shot micro-benchmark shows neither).
- **It's a fair, honest scoreboard, not a victory lap.** On some problems a
  dedicated convex solver (e.g. CVXPY on max-Sharpe) is competitive or faster per
  solve — the tables show it plainly. The point is that jaxfolio is now in the
  same performance class as the specialized libraries while keeping its
  differentiable, unified-interface design.

## Extending

Adding another library is one small class in `libraries.py`: implement
`is_available()` and `solve(problem, ctx)`, returning weights aligned to
`ctx.assets`. Raise `Unsupported` for problems the library doesn't natively
handle. Add the class to `ALL_ADAPTERS` and it joins every table and chart.
