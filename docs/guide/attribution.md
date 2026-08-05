# Explainable optimization

A convex optimizer normally tells you *what* to hold and nothing about *why*. If
an asset comes back at 0%, there is no way to ask whether it was excluded on its
own merits or crowded out by a limit elsewhere in the portfolio.

`jf.explain` answers that:

```python
import jaxfolio as jf

res = jf.minimum_variance(returns, constraints=[
    jf.GroupCap("tech", ["AAPL", "MSFT", "NVDA"], max=0.30),
])
print(jf.explain(res).explain_text())
```

```
Constraint attribution — Minimum Variance (9 assets)
  quality: exact  ·  stationarity 2.72e-04
  cost of capital (budget): +0.00010 per unit

Asset NVDA: 0.00%   [at lower]
  binding: tech (group_cap)   shadow price -0.00009
  own merit -0.00009  ·  charged by tech +0.00009  ·  net +0.00000

Asset MSFT: 23.50%
  cause: own merits (interior)
```

Read the third line: NVDA's *own merit* is negative — it wants in — but the `tech`
cap **charges** it enough to push the net over the line. It was not excluded
because it is unattractive; it was excluded because its sector was full. That is
the distinction the report exists to draw.

## Where the numbers come from

No second solve, and no extra solver. Every optimizer here minimizes `f` over a
convex set `C` by projected gradient, so at a solution `w*` stationarity says `w*`
is a *fixed point* of the projection:

$$w^* = P_{\mathcal C}(w^* - s\,\nabla f(w^*)) \qquad \text{for all small } s>0$$

So projecting `w* - s∇f` and keeping the projection's **own** multipliers hands
back the multipliers of the portfolio problem, divided by `s`:

| Recovered | From the projection | Meaning |
| --- | --- | --- |
| $\lambda$ | budget multiplier / `s` | cost of capital |
| $\theta_k$ | row multiplier / `s` | shadow price of named row `k` |
| $rc_i$ | $-\beta_i / s$ | reduced cost of asset `i` |

Total cost: one gradient and two projections, both **outside** the solver loop. It
is therefore solver-agnostic — which is how it retrofits a real KKT measure onto
the optax solvers, whose own `residual` is a weight-update norm and
[explicitly not an optimality test](optimizers.md#choosing-a-solver).

## The pricing decomposition

The reduced cost splits into the two terms a portfolio manager would name:

$$rc_i = \underbrace{g_i + \lambda}_{\text{intrinsic}} + \underbrace{\theta_{k(i)}}_{\text{imposed}}$$

At an active bound $rc_i \ne 0$, and "what put this asset here?" is answered by
whichever **term** is largest.

!!! warning "Rank the terms, not the multipliers"
    The obvious rule — take the largest multiplier among the constraints touching
    asset `i` — is wrong in a way that looks right. At $w_i = 0$ the box
    multiplier *equals the entire reduced cost*, so it dominates every group term
    by construction and `box_lower` would win every single time. "binding: tech"
    could never fire. Ranking the terms of the decomposition is the textbook LP
    column-pricing identity, and it degrades correctly: with no groups every zero
    attributes to `own`, which is the truthful answer.

## Signs

`shadow_price` is $\partial(\text{named objective}) / \partial(\text{active limit})$
— the derivative with respect to the bound that is actually binding. The sign
follows from `sense`, and the table below is worth reading carefully, because this
is where a reader will (reasonably) suspect an error:

| Objective | Active bound | `multiplier` $\theta$ | `shadow_price` | Reading |
| --- | --- | --- | --- | --- |
| minimize (variance, CVaR) | cap | $>0$ | $-\theta < 0$ | raising the cap **lowers** variance |
| minimize | floor | $<0$ | $-\theta > 0$ | raising the floor **raises** variance |
| maximize (Sharpe, utility, growth) | cap | $>0$ | $+\theta > 0$ | raising the cap **raises** Sharpe |
| maximize | floor | $<0$ | $+\theta < 0$ | raising the floor **lowers** Sharpe |

The invariant that never breaks: **relaxing** a constraint can only ever weakly
improve the optimum. Only the presentation flips.

!!! note "Units are per period, and are not annualized"
    Every other diagnostic on a `PortfolioResult` (`expected_return`,
    `volatility`, `sharpe`) is annualized. Multipliers are **not**, deliberately:
    ×252 is right for a variance objective (linear in the covariance) and wrong
    for Sharpe (√252), so scaling them would be wrong in a different way for each
    method. They are reported in the objective's own per-period units.

## Diagnostics

`explain()` returns a [`ConstraintReport`](../reference/optimizers.md). A digest
also lands in `result.metadata`, JSON-serializable, for consumers that never call
`explain`:

| Key | Meaning |
| --- | --- |
| `kkt_residual` | $\lVert d\rVert$, the part of the gradient the constraints could not absorb. Zero at an exact optimum. |
| `stationarity` | `kkt_residual / ‖∇f‖` — the same thing, scale-free. |
| `dual_quality` | `exact` / `approximate` / `unreliable`. See below. |
| `n_binding_constraints`, `binding_constraints` | Which named rows are actually moving the optimum. |

Per-constraint and per-asset views come back as Polars frames:

```python
report = jf.explain(res)
report.constraints_frame()   # which constraint costs the portfolio the most
report.to_frame()            # what put each asset where it is
report.binding()             # just the rows that move the optimum
report.for_asset("NVDA")
```

Constraint `status` is four-way, not two:

| Status | Meaning |
| --- | --- |
| `binding` | Active *and* carrying a non-zero multiplier — relaxing it moves the optimum. |
| `weakly_active` | On its bound with a **zero** multiplier (a degenerate vertex). Relaxing it will not move anything, and it is never named as a cause. |
| `inactive` | Slack. The multiplier is exactly zero, not merely small. |
| `unidentified` | Active, but the multiplier is not pinned by the data. `shadow_price` is `None`. |

## Five renderings, one report

The same fields, formatted for whoever is reading. None of them recompute anything.

| Call | For |
| --- | --- |
| `explain_text(top_n=…)` | prose — reading *one* asset's story |
| `to_table(max_assets=…, notes=…)` | a printed report: header block, constraints table, assets table, notes |
| `to_json(indent=…)` / `to_dict()` | a machine — the whole report, serialized |
| `to_frame()` / `constraints_frame()` | Polars — sorting, filtering, joining |
| `result.metadata` | a backtest log — the five-key digest, always present |

`to_table()` is the one to print. Every string **jaxfolio wrote** is folded to ASCII on
the way in — a report ends up in a log file, a CI artifact or a Windows console, and
`<=` survives an encoding that `≤` does not. The reasons and warnings are written for
this guide, so they arrive full of em dashes; the fold is a transliteration, not a
filter, so nothing a reader needs is lost. Your *own* strings — asset names, constraint
names — pass through verbatim, because mangling a ticker to fit a charset is the worse
trade:

```
==============================================================================
CONSTRAINT ATTRIBUTION -- Minimum Variance
==============================================================================
assets           9
objective        variance (minimize)
dual quality     exact
stationarity     2.72e-04
KKT residual     6.43e-08
cost of capital  +9.622e-05 per unit
binding rows     1 of 3: tech

CONSTRAINTS
name        kind       activity      bound  slack  multiplier  shadow price  status
----------  ---------  --------  ---------  -----  ----------  ------------  --------
tech        group_cap    30.00%  <= 30.00%  0.00%  +8.900e-05    -8.900e-05  binding
energy      group_row    45.29%  <= 50.00%  4.71%  +0.000e+00    +0.000e+00  inactive
financials  group_row    24.71%  >= 15.00%  9.71%  +0.000e+00    +0.000e+00  inactive

ASSETS
asset     weight  status     own merit     imposed         net  cause             confidence
--------  ------  --------  ----------  ----------  ----------  ----------------  ----------
ASSET_08   0.00%  at lower  -8.659e-05  +8.900e-05  +2.412e-06  tech (group_cap)  unique
ASSET_01  23.50%  interior         n/a         n/a  +0.000e+00  own merits        unique
...

NOTES
- multipliers are per period, in the objective's own units (not annualized)
```

Three details in there are load-bearing. The `bound` column names only a limit that
*could* bind — a cap at or above the budget and a floor at zero are not drawn, because a
column of `<= 100%` reads as a limit doing work. Anything the report declines to quote
prints as `n/a`, never as a `0.00` that would read as "costs nothing". And `max_assets`
announces what it dropped (`(5 further asset(s) not shown)`) rather than letting a
truncated report look complete.

`to_json()` carries the same discipline into the serialized form — a suppressed shadow
price is `null`:

```python
payload = jf.explain(res).to_dict()   # or .to_json(indent=2)
payload["constraints"][0]
# {'name': 'tech', 'kind': 'group_cap', 'activity': 0.30000002682209015,
#  'lower': 0.0, 'upper': 0.3, 'slack': -2.682209016002801e-08,
#  'multiplier': 8.900064131012186e-05, 'shadow_price': -8.900064131012186e-05,
#  'status': 'binding', 'members': ['AAPL', 'MSFT', 'NVDA']}
```

Top-level keys: `schema`, `method`, `objective`, `sense`, `available`, `quality`,
`reason`, `diagnostics`, `budget`, `constraints`, `assets`, `warnings`. Every scalar is a
plain Python `float` / `str` / `bool` / `None` — no numpy types, no `NaN`, so
`json.dumps(..., allow_nan=False)` holds. `schema` is versioned
(`jaxfolio.constraint_report/1`) so a consumer can tell layouts apart if this grows.

## What this cannot tell you

The list is not short, and every entry is enforced rather than merely documented.
When a number cannot be stood behind it comes back `None`.

**Non-smooth objectives — `min_cvar` refuses outright.** The CVaR objective
contains `max(loss - tau, 0)`, and at the optimum `tau` sits *exactly on a sample
loss*, generically. `jax.grad` returns one arbitrary subgradient at that kink, so
stationarity need not hold even at the exact solution. Quality is `unreliable` and
every shadow price is `None`. The primal active set is still reported, along with
`aux_stationarity` — a free bonus diagnostic, `|∂CVaR/∂tau|`, saying how far the
VaR threshold is from its own optimum.

**Unconverged solves.** Judged against the tolerance you asked for, plus a
scale-free backstop. `solver="sgd", max_iter=5` earns `unreliable` and gets no
shadow prices — and note that this is detected by a KKT residual computed
*independently* of the solver's own convergence measure, which for the optax
family cannot reveal it.

**Scale-invariant objectives have no cost of capital.** Sharpe and the
diversification ratio are degree-0 homogeneous, so by Euler's identity
$g'w^* = 0$ and "relax the budget" is meaningless — the optimum is unchanged up to
scaling. `budget_shadow_price` is `None` with a note. Detected **numerically**
(testing $|g'w|$), not from a list of method names, so a custom scale-invariant
objective gets the same protection. Row and box multipliers stay meaningful.

**Unidentified multipliers.** A row's multiplier is pinned only when at least one
member is *strictly interior* to its box — that member's stationarity equation is
what separates $\theta_k$ from $\beta_i$. When every member sits on a bound, only
the sum is determined and any split is an equally valid KKT certificate. The row
reports `status="unidentified"`, `shadow_price=None`, and every affected asset gets
`confidence="ambiguous"` with `primary=None`. The same holds for the budget
multiplier, which additionally needs an interior asset *outside* every binding row.

**`l2_reg` muddies the picture, and says so.** With a penalty active the
multipliers price the *penalized* objective, and more insidiously L2 softens
corners — so a constraint doing real economic work can read as `inactive`. A
warning is attached whenever `l2_reg > 0`.

**Non-convex objectives are local only.** `maximum_sharpe` and
`maximum_diversification` as written are not convex. The multipliers remain valid
on the local branch, so a warning is attached and no global-optimality claim is
made.

**Optimizers with no duals at all.** `equal_weight`, `inverse_volatility`,
`risk_parity`, the graph methods (HRP, HERC, MST) and the learning policies solve
no constrained program. `explain()` still returns a well-formed report, with
`available=False` and a *specific* reason. These are **feasible by construction**,
not **optimal subject to** — a zero weight is caused by the construction, not by a
constraint. `risk_parity`'s reason points at its existing
`metadata["risk_contributions"]`, which is the right per-asset attribution for
that method.

**And finally: a shadow price is not a forecast.** It is
$\partial f^*/\partial c$ on *in-sample estimated moments*. It says what the
optimizer would have done with a different limit, not what the market will do.

## Cost, and switching it off

One gradient and two projections per solve, outside the solver loop — negligible
next to the solve itself, and it never touches the weights (there is a test
asserting bit-identical results with the capture on and off). For tight backtest
loops where the explanation is never read:

```python
jf.OptimizerConfig(attribution=False)
```

`explain()` then returns `available=False` rather than failing — it never *requires*
the payload.

## Accuracy

Shadow prices are validated against central finite differences on re-solves of the
optimizer itself — no external solver involved — across all three sign regimes:

| Case | Agreement with finite difference |
| --- | --- |
| minimize + cap (variance vs a sector cap) | 4–5 significant figures |
| maximize + cap (Sharpe vs a sector cap) | 4–5 significant figures |
| minimize + floor (variance vs a sector floor) | 4 significant figures |

The floor case is included deliberately: taking an absolute value anywhere in the
sign handling inverts it while every cap test keeps passing.
