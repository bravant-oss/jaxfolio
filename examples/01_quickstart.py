"""Quickstart: optimize a synthetic portfolio three ways and plot the weights.

Run with:
    uv run python examples/01_quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import jaxfolio as jf
from jaxfolio import viz

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    # Reproducible, offline synthetic data.
    returns = jf.generate_returns(n_assets=10, n_days=756, seed=7)

    methods = {
        "Maximum Sharpe": jf.maximum_sharpe(returns),
        "Hierarchical Risk Parity": jf.hierarchical_risk_parity(returns),
        "Risk Parity (ERC)": jf.risk_parity(returns),
    }

    for name, result in methods.items():
        print(f"\n{name}")
        print(f"  Sharpe (annualized): {result.sharpe:.2f}")
        for asset, w in list(result.top(3).items()):
            print(f"    {asset}: {w:.1%}")

    # Plot the max-Sharpe allocation and the efficient frontier.
    frontier = viz.plot_efficient_frontier(returns, highlight=methods)
    viz.save(frontier, str(OUT / "quickstart_frontier.png"))
    print(f"\nSaved frontier -> {OUT / 'quickstart_frontier.png'}")


if __name__ == "__main__":
    main()
