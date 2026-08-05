"""Multi-period, cost-aware optimization: an execution trajectory, not a target.

Shows the three things worth knowing:
  * the glide path — quadratic impact spreads a large trade over several periods,
  * why a purely linear cost does *not* (it creates a no-trade region instead),
  * what path-awareness buys end to end, in a backtest against the myopic solve.

    uv run python examples/06_multiperiod_costs.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

import jaxfolio as jf
from jaxfolio import viz
from jaxfolio.backtest import compare, metrics_table

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

HORIZON = 10
GAMMA = 3.0
IMPACT_BPS = 100.0


def main() -> None:
    returns = jf.generate_returns(n_assets=8, n_days=500, seed=42)
    assets = jf.asset_columns(returns)
    n = len(assets)

    # A deliberately awkward starting book: everything in the single name the
    # myopic optimizer wants least, so there is a large unwind to schedule. The
    # myopic solve would do all of it in one trade, on day one.
    target = jf.mean_variance(returns, risk_aversion=GAMMA).weights
    worst = int(np.argmin(target))
    holdings = np.zeros(n)
    holdings[worst] = 1.0
    print(
        f"Starting book: 100% {assets[worst]}, which the myopic target puts at "
        f"{target[worst]:.4f}\n"
    )

    # --- 1. The trajectory ---------------------------------------------------- #
    print("=" * 72)
    print(f"1. The glide path (10bps spread + {IMPACT_BPS:.0f}bps quadratic impact)")
    print("=" * 72)

    res = jf.multi_period_mean_variance(
        returns,
        horizon=HORIZON,
        w_prev=holdings,
        risk_aversion=GAMMA,
        costs=jf.TradingCosts(spread_bps=10.0, impact_bps=IMPACT_BPS),
    )

    path = (
        pl.DataFrame(dict(zip(assets, res.trajectory.T, strict=True)))
        .with_columns(pl.Series("period", [f"period {t + 1}" for t in range(HORIZON)]))
        .select("period", *assets)
    )
    print("\nPlanned weight path:")
    print(path.select("period", pl.col(assets).round(4)))
    print(f"\nTrade size per period: {np.round(res.metadata['turnover_path'], 4).tolist()}")
    print(f"Total planned turnover: {res.metadata['total_turnover']:.4f}")
    print(f"Cost of the whole plan: {res.metadata['total_cost'] * 1e4:.2f} bps of NAV")
    first_row_matches = np.allclose(res.weights, path.select(assets).row(0))
    print("\nweights == row 0 (what to trade into today):", first_row_matches)
    print("terminal (long-run) target:", np.round(res.metadata["terminal_weights"], 4).tolist())
    print(f"smoothing bias: {res.metadata['smoothing_bias']:.3e} (audits the L1 surrogate)")

    # --- 2. Linear cost alone does not spread execution ---------------------- #
    print()
    print("=" * 72)
    print("2. Why impact matters: linear cost alone gives a no-trade region, not a path")
    print("=" * 72)
    print(f"\n{'costs':<32} {'turnover, first 6 periods':<38} {'total':>7}")
    for label, spread, impact in (
        ("frictionless", 0.0, 0.0),
        ("50bps spread only", 50.0, 0.0),
        ("10bps spread + 100bps impact", 10.0, 100.0),
        ("10bps spread + 500bps impact", 10.0, 500.0),
    ):
        r = jf.multi_period_mean_variance(
            returns,
            horizon=HORIZON,
            w_prev=holdings,
            risk_aversion=GAMMA,
            costs=jf.TradingCosts(spread_bps=spread, impact_bps=impact),
        )
        per_period = " ".join(f"{x:5.3f}" for x in r.metadata["turnover_path"][:6])
        print(f"{label:<32} {per_period:<38} {r.metadata['total_turnover']:>7.3f}")
    print("\nNote how the spread-only row front-loads everything into period 1: a")
    print("proportional cost makes you trade *less*, but gives no reason to trade")
    print("*later*. Only the quadratic impact term schedules execution.")

    # --- 3. End to end, against the myopic optimizer ------------------------- #
    print()
    print("=" * 72)
    print("3. Backtest: myopic vs path-aware under a punitive 50bps cost")
    print("=" * 72)

    results = compare(
        returns,
        {
            "myopic": partial(jf.mean_variance, risk_aversion=GAMMA),
            "multi-period": partial(
                jf.multi_period_mean_variance,
                horizon=HORIZON,
                risk_aversion=GAMMA,
                costs=jf.TradingCosts(spread_bps=50.0, impact_bps=500.0),
            ),
        },
        lookback=252,
        rebalance_every=21,
        transaction_cost=0.005,
    )
    table = metrics_table(results)
    print()
    summary = table.select(
        "strategy", "annual_return", "sharpe", "avg_turnover", "total_cost"
    ).with_columns(pl.exclude("strategy").round(4))
    print(summary)

    drag = results["myopic"].metrics["total_cost"] - results["multi-period"].metrics["total_cost"]
    print(f"\nCost drag avoided: {drag:.4f} ({drag * 1e4:.0f} bps of cumulative NAV)")

    # --- 4. Figures ----------------------------------------------------------- #
    # The dedicated multi-period plots read the trajectory and its diagnostics
    # directly, so they label periods (not dates) and anchor at the starting book.
    regimes = {
        "50bps spread only": jf.TradingCosts(spread_bps=50.0),
        "10bps + 100bps impact": jf.TradingCosts(spread_bps=10.0, impact_bps=100.0),
        "10bps + 500bps impact": jf.TradingCosts(spread_bps=10.0, impact_bps=500.0),
    }
    comparison = {
        label: jf.multi_period_mean_variance(
            returns, horizon=HORIZON, w_prev=holdings, risk_aversion=GAMMA, costs=cost
        )
        for label, cost in regimes.items()
    }

    viz.save(viz.plot_weight_path(res), str(OUT / "multiperiod_glidepath.png"))
    # ``target`` is the myopic optimum — a genuine external reference for what a
    # single rebalance would have cost. The plan's own terminal weights are not:
    # for a monotone path, total turnover equals |w_T - w_prev| identically.
    viz.save(
        viz.plot_turnover_schedule(res, reference_weights=target),
        str(OUT / "multiperiod_schedule.png"),
    )
    viz.save(viz.plot_cost_comparison(comparison), str(OUT / "multiperiod_costs.png"))
    viz.save(
        viz.multiperiod_dashboard(res, comparison=comparison, reference_weights=target),
        str(OUT / "multiperiod_dashboard.png"),
    )
    viz.save(viz.dashboard(results, returns), str(OUT / "multiperiod_backtest.png"))
    print(f"\nWrote figures to {OUT}/")


if __name__ == "__main__":
    main()
