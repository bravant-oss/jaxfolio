# Changelog

All notable changes to **jaxfolio** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This file is generated with [git-cliff](https://git-cliff.org) from
[Conventional Commits](https://www.conventionalcommits.org) and curated for clarity.

## [Unreleased]

### Features

- **Explainable optimization** — `jf.explain(result)` reports which constraints
  bind, what each one costs, and *why* each asset holds the weight it does. A 0%
  allocation now comes with an attribution: "excluded on its own merits" versus
  "its sector was full", with the shadow price of the constraint responsible.
  The multipliers are recovered from the projection itself, not from a second
  solve: at a solution the weights are a fixed point of the projection, so
  projecting `w* - s*grad f` and keeping the projection's own multipliers returns
  the portfolio problem's multipliers divided by `s`. One gradient and two
  projections, outside the solver loop, and solver-agnostic — which incidentally
  retrofits a genuine KKT residual onto the optax solvers, whose own `residual` is
  a weight-update norm and explicitly not an optimality test.
  Attribution ranks the *terms* of the reduced-cost decomposition
  (`rc = intrinsic + imposed`), not the multipliers. Ranking multipliers looks
  right and is wrong: at a zero weight the box multiplier equals the whole reduced
  cost, so it would dominate every group term by construction and "binding: tech"
  could never fire.
  Shadow prices are validated against central finite differences on re-solves of
  the optimizer itself, to 4-5 significant figures, in all three sign regimes
  (minimize+cap, maximize+cap, minimize+floor).
  **The report refuses rather than guesses.** `min_cvar` gets no shadow prices at
  all — its objective is non-smooth exactly where the optimum sits, so `jax.grad`
  returns an arbitrary subgradient and stationarity need not hold even when exact.
  Unconverged solves are suppressed. Scale-invariant objectives (max-Sharpe,
  max-diversification) report no cost of capital, detected numerically via Euler's
  identity rather than from a list of method names. Multipliers that are not pinned
  by the data — a binding row whose every member sits on a bound — are reported as
  `unidentified` with the affected assets marked `ambiguous`. `l2_reg` and
  non-convexity attach warnings. Optimizers that solve no constrained program
  (`equal_weight`, `inverse_volatility`, `risk_parity`, the graph and learning
  methods) return a well-formed report with `available=False` and a specific
  reason — `risk_parity`'s points at its own `risk_contributions`.
  Adds the optional `PortfolioResult.attribution` field (following the
  `trajectory` precedent) plus a small JSON-serializable digest in `metadata`, and
  `OptimizerConfig(attribution=False)` to opt out in tight backtest loops.
  Capturing the diagnostics never perturbs the weights — asserted bit-identical.
  A report renders five ways, for five readers: `explain_text()` as prose,
  `to_table()` as a printed report (header block, constraints table, assets table,
  notes — deliberately pure ASCII so it survives a log file, a CI artifact and a
  non-UTF-8 console), `to_dict()` / `to_json()` serialized whole, the pandas frames,
  and the `metadata` digest. The refusals survive both new formats: a suppressed
  shadow price prints as `n/a` and serializes as `null`, never as a `0.0` that would
  read as "this constraint costs nothing"; `to_table(max_assets=…)` announces what it
  dropped rather than letting a truncated report look complete; and the serialized
  form carries plain Python scalars only (no numpy types, no `NaN`) under a versioned
  `schema` key.
- **Named constraints** — sector and group caps that carry an identity. Where
  `long_only` / `weight_bounds` shape the answer anonymously, `GroupCap("tech",
  [...], max=0.30)`, `GroupFloor`, `Box` (now accepting per-asset bound vectors)
  and `Budget` can be *reported on*, which is the groundwork for attributing a 0%
  allocation to the constraint that caused it. Accepted by every optimizer that
  solves a constrained program, via `constraints=[...]` or
  `OptimizerConfig(constraints=...)`.
  A cap is a **guarantee**, not a preference: it is enforced by an exact Euclidean
  projection — a nested bracketed bisection on the constraint multipliers, in the
  same exactness class as the existing box projection — rather than by a penalty
  that lands "near" feasible. Validated against CVXPY over 40 random problems:
  weights to 2.1e-8, and the recovered Lagrange multipliers to 3.9e-9.
  Two deliberate restrictions, both structural rather than temporary. **Groups must
  partition the universe** (each asset in at most one group): disjoint supports are
  exactly what decouples the row multipliers and keeps the projection exact, and
  the SPG solver's stopping test only certifies KKT stationarity for an exact
  projection — an alternating projection would break it *silently* inside
  `lax.while_loop`. And the closed-form weightings (`risk_parity`,
  `equal_weight`, `inverse_volatility`, the graph methods) plus
  `multi_period_mean_variance` **raise rather than ignore** a constraint they
  cannot honor. Infeasible sets are rejected before any solve, by a test that is
  necessary *and* sufficient, with the offending constraint named and the gap
  quantified.
  Not expressible: overlapping or nested groups, rows with coefficients other than
  1 (beta/factor limits), gross-exposure caps, cardinality and minimum position
  size. See the [constraints guide](docs/guide/constraints.md).
  Inert for every existing call — no named constraints means the identical solver
  kernel, the identical jit cache entry, and bit-for-bit identical weights,
  verified across all 14 optimizers.
- **Multi-period ("path-aware") optimization** via the new
  `multi_period_mean_variance`. Where every existing optimizer answers "what
  should I hold?" from a snapshot, this one answers "what should I *do* today?":
  it solves for a whole `(T, N)` weight path at once, anchored at the portfolio
  you currently hold, with bid-ask spread, proportional fees, and quadratic market
  impact priced *inside* the objective. `result.weights` is the first row — the
  trade to place now — and `result.trajectory` is the full plan. Forecasts default
  to one `(mu, Sigma)` held across the horizon, with optional `mu_path` /
  `cov_path` term structures.
  Note that **quadratic impact is what produces a glide path**: a purely linear
  cost creates a no-trade region but gives no reason to split a trade, so the
  optimum jumps once and holds (this matches an exact CVXPY QP, and is not an
  artifact). Turnover control is deliberately **soft** — costs are priced, not
  capped; a hard cap couples the periods and is out of scope.
- Added `jaxfolio.TradingCosts`, a frozen cost spec in bps of traded notional per
  unit turnover (scalar or per-asset), plus the optional
  `PortfolioResult.trajectory` field for weight paths.
- Five dark-themed multi-period plots in `jaxfolio.viz`: `plot_weight_path`,
  `plot_turnover_schedule`, `plot_path_convergence`, `plot_cost_comparison`, and
  the composite `multiperiod_dashboard`. They read a trajectory and its
  diagnostics directly — labelling integer periods rather than dates and anchoring
  period 0 at the book the plan starts from — and fall back to per-asset lines when
  a path shorts, since stacked bands cannot represent signed exposures.
  The "one-shot rebalance" reference is opt-in via `reference_weights=`: a monotone
  path's total turnover equals `|w_T - w_prev|` *identically*, so drawing the plan's
  own terminal weights as a reference would look like a comparison while measuring
  nothing.
- `backtest()` gained `holdings_aware` and `follow_trajectory`. An optimizer that
  declares a `w_prev` keyword is now handed the live portfolio at each rebalance,
  and a returned `trajectory` is executed step by step over the following periods
  instead of being discarded. Both are auto-detected and **inert for every
  existing optimizer** — no behavior change for any current strategy. Also adds a
  `total_cost` metric; `avg_turnover` is now documented as per *trading* period,
  which is identical to per-rebalance for path-less optimizers.
- The shared projected-gradient solver is now rank-agnostic, so a `(T, N)` weight
  path is a first-class variable, and `solve_constrained` /
  `solve_projected_gradient` additionally report `info["residual"]`. Verified
  bit-for-bit identical results for all existing single-period optimizers.
- **`OptimizerConfig.solver` now accepts any optax optimizer**, not just
  `"spg"` / `"adam"` — by name (`solver="adamw"`, `"sgd"`, `"rmsprop"`, ...,
  resolved against `optax` and `optax.contrib`) or by factory
  (`solver=optax.adamw`). Hyperparameters go in the new `solver_options`
  (`{"weight_decay": 1e-3}`). The solver travels as a hashable
  `jaxfolio.solvers.SolverSpec`, so the jit cache still hits across repeated
  solves; pre-built transformations (`optax.adam(1e-2)`) are rejected with a
  pointer to the factory spelling, because they would recompile every solve.
- Added `jaxfolio.solvers` (`resolve_solver`, `build_optimizer`,
  `available_solvers`, `SolverSpec`), re-exported from `jaxfolio.toolkit`;
  `jf.available_solvers()` lists the selectable names.
- `deep_sharpe` gained `optimizer` / `optimizer_options`, so the MLP allocation
  policy can be trained with any optax optimizer instead of only Adam.
- `OptimizerConfig` now validates `solver` / `solver_options` at construction,
  with close-match suggestions for typos, and rejects line-search optimizers
  (`optax.lbfgs`, `optax.polyak_sgd`) with an actionable message — they need
  per-step `value`/`grad`/`value_fn` the projected-gradient loop cannot supply.

### Fixed

- `CustomStrategy.from_objective` now forwards `config.solver` /
  `config.solver_options` to the solver. It previously dropped them, so a
  strategy configured with `solver="adam"` silently ran SPG — and the supplied
  `learning_rate` was reinterpreted as SPG's initial Barzilai-Borwein step.

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
