"""Backtest every optimizer family head-to-head and save a dark dashboard.

Run with:
    uv run python examples/02_method_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import jaxfolio as jf
from jaxfolio import viz
from jaxfolio.backtest import compare, metrics_table

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    returns = jf.generate_returns(n_assets=12, n_days=1000, seed=11)

    optimizers = {
        "1/N": jf.equal_weight,
        "Min Variance": jf.minimum_variance,
        "Max Sharpe": jf.maximum_sharpe,
        "Risk Parity": jf.risk_parity,
        "HRP": jf.hierarchical_risk_parity,
        "MST Centrality": jf.mst_centrality,
        "Online EG": jf.online_gradient,
    }

    print("Running walk-forward backtests...")
    results = compare(
        returns, optimizers, lookback=252, rebalance_every=21, transaction_cost=0.0010
    )

    table = metrics_table(results)[
        ["annual_return", "annual_volatility", "sharpe", "max_drawdown", "avg_turnover"]
    ]
    print("\n=== Backtest metrics ===")
    print(table.round(3).to_string())

    # Full-in-sample optimized portfolios for the frontier overlay.
    highlight = {
        "Max Sharpe": jf.maximum_sharpe(returns),
        "Min Variance": jf.minimum_variance(returns),
        "HRP": jf.hierarchical_risk_parity(returns),
    }

    # Composite dashboard.
    dash = viz.dashboard(results, returns, highlight=highlight)
    viz.save(dash, str(OUT / "comparison_dashboard.png"))

    # Individual plots.
    viz.save(viz.plot_equity_curves(results), str(OUT / "equity_curves.png"))
    viz.save(viz.plot_drawdown(results), str(OUT / "drawdown.png"))
    viz.save(viz.plot_correlation_network(returns), str(OUT / "correlation_network.png"))
    viz.save(
        viz.plot_dendrogram(jf.hierarchical_risk_parity(returns)),
        str(OUT / "hrp_dendrogram.png"),
    )
    viz.save(viz.plot_risk_contributions(jf.risk_parity(returns)), str(OUT / "risk_contrib.png"))

    print(f"\nSaved dashboards and plots -> {OUT}/")


if __name__ == "__main__":
    main()
