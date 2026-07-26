"""Named constraints: caps that are guarantees, and that you can reason about.

Shows the four things worth knowing:
  * a named sector cap binds *exactly*, and reshapes the portfolio around it,
  * what the cap costs — the objective gap against the unconstrained optimum,
  * the recovered Lagrange multipliers, i.e. the shadow price of each constraint,
  * how infeasible sets are rejected up front, with the gap quantified.

    uv run python examples/07_constraints.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jaxfolio as jf
from jaxfolio import viz
from jaxfolio.constraints import InfeasibleConstraints, compile_constraints

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

TECH_CAP = 0.30


def _exposure(weights, assets, members) -> float:
    idx = [assets.index(m) for m in members]
    return float(np.asarray(weights)[idx].sum())


def main() -> None:
    returns = jf.generate_returns(n_assets=9, n_days=500, seed=42)
    assets = list(returns.columns)

    free = jf.minimum_variance(returns)

    # Three disjoint "sectors" — the partition the projection requires. On
    # synthetic data the labels are arbitrary, so put the three names the
    # unconstrained optimizer likes *most* into "tech": otherwise the cap sits
    # above the natural exposure and the whole demonstration is vacuous.
    ranked = [assets[i] for i in np.argsort(-np.asarray(free.weights))]
    tech, energy, financials = ranked[:3], ranked[3:6], ranked[6:]

    # --- 1. A named cap binds exactly ------------------------------------- #
    print("=" * 72)
    print("1. A sector cap is a guarantee, not a preference")
    print("=" * 72)
    print(f"tech       = {', '.join(tech)}")
    print(f"energy     = {', '.join(energy)}")
    print(f"financials = {', '.join(financials)}\n")

    capped = jf.minimum_variance(returns, constraints=[jf.GroupCap("tech", tech, max=TECH_CAP)])

    print(f"{'asset':<12} {'unconstrained':>14} {'tech<=30%':>12} {'delta':>9}")
    for i, name in enumerate(assets):
        tag = " (tech)" if name in tech else ""
        print(
            f"{name:<12} {free.weights[i]:>14.4f} {capped.weights[i]:>12.4f} "
            f"{capped.weights[i] - free.weights[i]:>+9.4f}{tag}"
        )
    print(
        f"\ntech exposure: {_exposure(free.weights, assets, tech):.4f} "
        f"-> {_exposure(capped.weights, assets, tech):.4f}  (cap {TECH_CAP})"
    )
    print(f"both fully invested: {free.weights.sum():.6f} / {capped.weights.sum():.6f}")

    # --- 2. What the cap costs -------------------------------------------- #
    print("\n" + "=" * 72)
    print("2. What the constraint costs")
    print("=" * 72)

    _mu, cov, _names, _ = jf.toolkit.moments(returns)
    cov = np.asarray(cov)
    var_free = float(free.weights @ cov @ free.weights)
    var_capped = float(capped.weights @ cov @ capped.weights)
    print(f"variance unconstrained : {var_free:.8f}")
    print(f"variance with tech<=30%: {var_capped:.8f}")
    print(
        f"cost of the cap        : {var_capped - var_free:+.8f} "
        f"({(var_capped / var_free - 1) * 100:+.2f}%)"
    )
    print(f"annualized volatility  : {free.volatility:.4f} -> {capped.volatility:.4f}")

    # --- 3. Shadow prices ------------------------------------------------- #
    print("\n" + "=" * 72)
    print("3. Shadow prices — the multipliers the projection already computes")
    print("=" * 72)

    import jax.numpy as jnp

    from jaxfolio.constraints.structured import project_grouped_duals

    compiled = compile_constraints(
        [
            jf.GroupCap("tech", tech, max=TECH_CAP),
            jf.GroupCap("energy", energy, max=0.50),
            jf.GroupCap("financials", financials, max=0.90),
        ],
        assets,
    )
    # Probe the projection at the *unconstrained* optimum: the multipliers price
    # how hard each row has to push to pull that point back into the feasible set.
    lower, upper, budget, gid, g_lo, g_hi = compiled.pparams()
    duals = project_grouped_duals(jnp.asarray(free.weights), lower, upper, budget, gid, g_lo, g_hi)
    print(f"{'constraint':<14} {'exposure':>10} {'limit':>8} {'multiplier':>12} {'status':>14}")
    for k, name in enumerate(compiled.row_names):
        members = compiled.members(name)
        theta = float(duals.rows[k])
        if not bool(duals.identified[k]):
            status = "not identified"
        elif theta == 0.0:
            status = "slack"
        else:
            status = "BINDING"
        print(
            f"{name:<14} {_exposure(duals.weights, assets, members):>10.4f} "
            f"{compiled.row_upper[k]:>8.2f} {theta:>+12.5f} {status:>14}"
        )
    print(
        "\nA zero multiplier on a slack row is exact, not a rounding artifact. "
        "'not identified'\nmeans every member sits on a bound, so the split between "
        "the row and the box is\narbitrary — the report declines to quote a number "
        "it cannot stand behind."
    )

    # --- 4. Infeasibility is caught before any solve ---------------------- #
    print("\n" + "=" * 72)
    print("4. Infeasible sets are rejected up front, with the gap quantified")
    print("=" * 72)

    attempts = [
        (
            "caps that cannot fund the budget",
            [
                jf.GroupCap("tech", tech, max=0.30),
                jf.GroupCap("energy", energy, max=0.25),
                jf.GroupCap("financials", financials, max=0.30),
            ],
        ),
        (
            "overlapping groups",
            [
                jf.GroupCap("tech", tech, max=0.30),
                jf.GroupCap("us_large", [tech[0], energy[0]], max=0.40),
            ],
        ),
        ("a typo in a ticker", [jf.GroupCap("tech", [tech[0], "NOPE_01"], max=0.30)]),
    ]
    for label, cons in attempts:
        try:
            jf.minimum_variance(returns, constraints=cons)
        except (InfeasibleConstraints, ValueError, KeyError) as exc:
            # KeyError stringifies with quotes; keep only the leading sentence.
            first = str(exc).strip("\"'").split(". ")[0].rstrip(".")
            print(f"\n{label}:\n  {type(exc).__name__}: {first}.")

    # --- Figures ----------------------------------------------------------- #
    fig = viz.plot_weights(capped, title=f"Minimum variance, tech capped at {TECH_CAP:.0%}")
    fig.savefig(OUT / "constraints_weights.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'constraints_weights.png'}")


if __name__ == "__main__":
    main()
