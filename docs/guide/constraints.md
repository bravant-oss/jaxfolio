# Named constraints

Every optimizer already takes `long_only` and `weight_bounds`. Those are
*anonymous*: they shape the answer but leave no trace in it. If an asset comes
back at 0%, nothing in the result says whether it was excluded on its own merits
or crowded out by a limit somewhere else.

Named constraints fix that. A constraint with a name can be reported on:

```python
import jaxfolio as jf

res = jf.minimum_variance(returns, constraints=[
    jf.GroupCap("tech", ["AAPL", "MSFT", "NVDA"], max=0.30),
    jf.Box(lower=0.0, upper=0.10),
])
```

The cap is a **guarantee**, not a preference — `tech` will hold at most 30.00%,
and the solver reaches the true constrained optimum rather than an approximately
feasible point nearby. That distinction is the whole design, and the rest of this
page explains what it costs and where it stops.

## The vocabulary

| Spec | Meaning |
| --- | --- |
| [`Box`](../reference/types.md) | Per-asset bounds `lower <= w_i <= upper`. Scalars, or per-asset sequences. |
| [`GroupCap`](../reference/types.md) | A named group's summed weight, capped (`max=`), floored (`min=`), or banded (both). |
| `GroupFloor` | Sugar for `GroupCap(min=...)`. Returns a `GroupCap`, so a group can never carry a floor and cap that disagree. |
| [`Budget`](../reference/types.md) | Total invested weight. Defaults to `1.0`; set `Budget(0.0)` for a dollar-neutral book. |

Constraints can arrive either as the `constraints=` keyword or on an
`OptimizerConfig` — the latter is what a backtester threads through:

```python
cfg = jf.OptimizerConfig(
    long_only=True,
    constraints=[jf.GroupCap("tech", ["AAPL", "MSFT"], max=0.30)],
)
res = jf.mean_variance(returns, config=cfg)
```

Passing both raises, rather than silently picking one.

`Box` composes predictably: an unscoped box **replaces** the bounds implied by
`long_only` / `weight_bounds`, while an `assets`-scoped box **intersects** with
whatever is already in force. So per-name overrides stack:

```python
constraints=[
    jf.Box(lower=0.0, upper=0.10),                              # everything at 10%
    jf.Box(upper=0.02, assets=["ILLIQ"], name="illiquid"),      # tighter on one name
]
```

`long_only=True` is a floor on the floor: it can only tighten, never licence a
short an explicit `Box` forbade.

## Groups must partition the universe

**Each asset may belong to at most one group.** Overlapping groups are rejected at
compile time:

```
ValueError: GroupCap('us_large') and 'tech' both constrain 'AAPL'.
Group constraints must partition the universe — each asset may belong to at most
one group. That disjointness is what makes the projection exact and the shadow
prices trustworthy; overlapping groups would need an alternating projection that
is inexact at any finite iteration count. Workaround: express one dimension as a
partition and price the other with l2_reg or a custom objective penalty.
```

This is structural, not an oversight. The projected-gradient solvers stop on
`||w - P(w - ∇f)||`, which certifies KKT stationarity **only if `P` is the true
Euclidean projection**, and the Barzilai-Borwein step assumes iterates from a
nonexpansive map. Disjoint supports are exactly what keeps the projection exact:
stationarity gives

$$w_i = \mathrm{clip}(v_i - \lambda - \theta_{k(i)},\ l_i,\ u_i)$$

and when each asset is touched by one row, $\theta_k$ **decouples given
$\lambda$** — each group's sum becomes an independent 1-D monotone function of its
own multiplier. The projection is then a fixed two-level nest of bracketed
bisections, with no tolerance to tune. Let supports overlap and the multipliers
couple; the projection needs Dykstra or ADMM, which is inexact at any finite
iteration count and would break the stopping test *silently* inside
`jax.lax.while_loop`.

This is the same trade the [multi-period guide](multiperiod.md) documents for
hard turnover caps, and for the same reason.

## Infeasibility is caught up front, exactly

Because the rows are disjoint, the reachable total is exactly the sum of each
row's reachable interval plus the ungrouped assets' box. The feasibility test is
therefore **necessary and sufficient** — never a false alarm, and a set that
compiles can always be satisfied:

```python
jf.minimum_variance(returns, constraints=[
    jf.GroupCap("tech", tech, max=0.30),
    jf.GroupCap("energy", energy, max=0.25),
    jf.GroupCap("financials", fins, max=0.30),
])
# InfeasibleConstraints: the constraints admit a total weight only in [0, 0.85],
# but the budget requires 1 — above the reachable range by 0.15. Rows: 'tech' in
# [0, 0.3], 'energy' in [0, 0.25], 'financials' in [0, 0.3] and every asset
# belongs to a group.
```

Validate a set without solving anything with
[`check_feasible`](../reference/toolkit.md):

```python
from jaxfolio.constraints import check_feasible
check_feasible(constraints, assets)   # raises, or returns None
```

Unknown tickers get a suggestion (`Did you mean 'AAPL'?`), and duplicate
constraint names are rejected so attribution can always refer to a row
unambiguously.

## Which optimizers accept them

Named constraints work with every optimizer that actually solves a constrained
program: `minimum_variance`, `mean_variance`, `maximum_sharpe`,
`maximum_diversification`, `kelly`, `min_cvar`, `black_litterman`, and any
[custom strategy](custom-strategies.md) built with `CustomStrategy.from_objective`.

The rest **raise rather than ignore**:

```python
jf.risk_parity(returns, constraints=[jf.GroupCap("tech", tech, max=0.3)])
# NotImplementedError: risk_parity is a closed-form weighting and cannot honor
# named constraints ('tech'). It has no optimization program for a cap to
# constrain. Use minimum_variance or mean_variance instead.
```

!!! warning "Silence would be worse"
    `risk_parity`, `equal_weight`, `inverse_volatility` and the graph methods
    (HRP, HERC, MST) build weights analytically — `risk_parity`'s log-barrier
    keeps every position strictly positive, so no bound is ever active and the
    budget is imposed by normalizing a fixed point rather than as a constraint.
    Accepting a cap and quietly returning weights that violate it is the one
    outcome worse than refusing, so they refuse.

`multi_period_mean_variance` also refuses, but for a different reason: the
row-wise projection *can* take per-period group caps — the feasible set stays a
Cartesian product over periods — but a cap on a *horizon aggregate* is not
expressible, and which of the two a user means is undecided. Rather than guess, it
raises.

## Performance

The grouped kernel is a nested bisection, so it costs more than a plain box
projection. Measured per projection call on CPU:

| Assets | Groups | Box | Grouped | Ratio |
| --- | --- | --- | --- | --- |
| 50 | 5 | 4.6 µs | 21.8 µs | 4.8× |
| 200 | 11 | 13.9 µs | 35.0 µs | 2.5× |
| 500 | 11 | 15.0 µs | 51.7 µs | 3.4× |

Each inner sweep is `O(n)` (a `segment_sum` over a group-id vector, not a dense
`(K, n)` mask), which is why the ratio *falls* as the universe grows.

**Solves with no named constraints are completely unaffected** — they take the
identical kernel, with the identical jit cache entry, and return bit-for-bit
identical weights.

### The jit cache

Every constraint *value* travels as a traced argument; only the projection kind
and the shapes `(n_assets, n_rows)` are static. So:

| Change | Recompiles? |
| --- | --- |
| A cap value, `0.30 → 0.32` | No |
| A per-asset bound vector | No |
| Which tickers are in a sector | No |
| The **number** of groups | Yes, one entry |
| The number of assets | Yes, one entry |

That is what makes an interactive relaxation cheap: perturbing a limit reuses the
already-compiled kernel.

## Limitation: what is not expressible

In the spirit of the multi-period guide's turnover section, the honest list.

**Overlapping or nested groups.** Structural, per above. A `"tech"` cap and a
`"US large-cap"` cap that share a ticker are rejected. *Workaround:* express one
dimension as a partition and price the other softly with `l2_reg` or a penalty
inside a custom objective. Strictly *nested* (laminar) families — sector within
region — do admit a recursive exact projection and are a plausible extension, but
are currently rejected alongside genuine overlap.

**Rows with coefficients other than 1.** A general `a'w <= c` for beta or
factor-exposure limits. The inner bisection generalizes readily, but the *outer*
budget bisection is only provably monotone for all-ones rows: with general
coefficients the group sum $\sum_i w_i$ is no longer pinned by the row, and
monotonicity of the total in $\lambda$ stops being obvious. Tracked rather than
guessed at.

**Gross-exposure / leverage caps** (`||w||_1 <= G`). Needs a third bisection level
and is only meaningful for long-short books. Tracked, not yet implemented.

**Cardinality and minimum position size** — "at most 20 names",
`w_i ∈ {0} ∪ [0.01, 0.10]`. Non-convex: there is no Euclidean projection onto a
non-convex set for a projected-gradient method to converge through. This needs a
MIQP and is not on the roadmap.

**Constraints on the CVaR auxiliary variable.** `min_cvar` optimizes a packed
`[w, tau]` vector; `tau` is unconstrained by construction, so a VaR-level
constraint is not expressible through the projection.

**Hard turnover caps.** Still soft — see [multi-period](multiperiod.md). Costs are
priced, not capped.

## Accuracy

The projection and its multipliers are validated against CVXPY (CLARABEL) as an
independent QP solver. Over 40 random problems (6–40 assets, 2–4 rows, long-only
and long-short) in float64:

| Quantity | Worst deviation |
| --- | --- |
| Weights | 2.1e-8 |
| Budget multiplier `λ` | 2.2e-9 |
| Row multipliers `θ` (identified) | 3.9e-9 |

!!! note "Why *identified*"
    A row's multiplier is pinned by the data only when at least one of its members
    is **strictly interior** to its own box — that member's stationarity equation
    is what separates $\theta_k$ from the box multiplier $\beta_i$. When every
    member sits on a bound, only the *sum* $\theta_k + \beta_i$ is determined, and
    any split of the shadow price between the row and the bounds is an equally
    valid KKT certificate. jaxfolio reports the minimum-norm choice
    ($\theta_k = 0$, everything attributed to the box), which is the reading users
    mean — but a different solver will legitimately report a different split. In
    the 40-problem sweep, all 14 unidentified rows disagreed with CVXPY, by as much
    as 0.37, while every identified row agreed to 4e-9.

    The same holds for the budget multiplier: it is pinned only by an interior
    coordinate that is *also* free of a binding row. The projection kernels expose
    `identified` and `budget_identified` flags precisely so a report can decline to
    quote a number it cannot stand behind.

The projections are also tested for **optimality**, not just feasibility, via the
variational inequality $\langle v - P(v),\ w - P(v)\rangle \le 0$ that defines a
Euclidean projection — the property the SPG stopping test depends on, and which
feasibility alone does not establish.
