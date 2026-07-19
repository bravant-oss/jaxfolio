"""Author and register a custom strategy, then backtest it against the built-ins.

Shows both authoring modes:
  * from_weights   — return a weight vector directly (a momentum tilt).
  * from_objective — supply a JAX objective and reuse the shared solver
    (entropy-regularized minimum variance).

    uv run python examples/05_custom_strategy.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxfolio as jf
from jaxfolio import toolkit as tk
from jaxfolio import viz
from jaxfolio.backtest import compare, metrics_table
from jaxfolio.custom import CustomStrategy, custom_strategy

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


# --- Mode 1: a weight function (12-period momentum, long-only) --------------- #
def momentum_weights(returns):
    """Weight proportional to trailing cumulative return, clipped at zero."""
    cum = (1.0 + returns).prod() - 1.0
    return np.clip(cum.to_numpy(), 0.0, None)


momentum = custom_strategy(
    "momentum", momentum_weights, register=True, description="Long-only momentum tilt"
)


# --- Mode 2: a JAX objective solved by the shared projected-gradient core ----- #
def entropy_minvar(w, ctx):
    """Minimum variance minus an entropy bonus (encourages diversification)."""
    variance = w @ ctx.cov @ w
    entropy = -jnp.sum(w * jnp.log(w + 1e-9))
    return variance - 0.002 * entropy


entropy_strategy = CustomStrategy.from_objective(
    "entropy_minvar", entropy_minvar, register=True, description="Entropy-regularized min variance"
)


def main() -> None:
    returns = jf.generate_returns(n_assets=12, n_days=1000, seed=11)

    # Custom strategies are registered and show up alongside the built-ins.
    print("Custom strategies in the registry:", jf.list_strategies(custom_only=True))
    print("Total strategies available:", len(jf.list_strategies()))

    # They run in compare() exactly like built-ins.
    results = compare(
        returns,
        {
            "Momentum (custom)": momentum,
            "Entropy MinVar (custom)": entropy_strategy,
            "Max Sharpe": jf.maximum_sharpe,
            "HRP": jf.hierarchical_risk_parity,
            "1/N": jf.equal_weight,
        },
        lookback=252,
        rebalance_every=21,
    )

    print("\n=== Backtest metrics ===")
    table = metrics_table(results)[["annual_return", "annual_volatility", "sharpe", "max_drawdown"]]
    print(table.round(3).to_string())

    highlight = {
        "Momentum (custom)": momentum(returns),
        "Entropy MinVar (custom)": entropy_strategy(returns),
        "Max Sharpe": jf.maximum_sharpe(returns),
    }
    dash = viz.dashboard(results, returns, highlight=highlight)
    viz.save(dash, str(OUT / "custom_dashboard.png"))
    print(f"\nSaved dashboard -> {OUT / 'custom_dashboard.png'}")

    # You can also build a custom strategy inline with the toolkit building blocks.
    mu, cov, names, _ = tk.moments(returns)
    print(
        f"\nToolkit exposes: moments(), make_projection(), solve_projected_gradient(), "
        f"finalize_result() — {len(names)} assets, mean vol "
        f"{float(jnp.sqrt(jnp.diag(cov)).mean()):.4f}"
    )


if __name__ == "__main__":
    main()
