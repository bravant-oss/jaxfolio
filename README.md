<p align="center">
  <img src="docs/logo.svg" alt="jaxfolio" width="120">
</p>

<h1 align="center">jaxfolio</h1>

<p align="center">
  Differentiable portfolio optimization & options strategies, powered by JAX.
</p>

<p align="center">
  <a href="https://bravant-oss.github.io/jaxfolio/"><img alt="docs" src="https://img.shields.io/badge/docs-jaxfolio-199e70?style=flat-square"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11+-3987e5?style=flat-square">
  <img alt="jax" src="https://img.shields.io/badge/JAX-9085e9?style=flat-square">
  <img alt="license" src="https://img.shields.io/badge/MIT-c98500?style=flat-square">
</p>

<br>

## Why jaxfolio

Portfolio construction has splintered into many methods — mean-variance, risk
parity, hierarchical clustering, learned policies — and, increasingly, options
overlays layered on top of an equity book. Each is powerful, but in practice they
arrive as **disconnected tools**: a QP solver here, a clustering script there, a
separate options pricer, each with its own inputs, quirks, and no common way to
compare them or hedge across them.

That fragmentation is the real cost. Swapping one strategy for another means
rewriting glue code; comparing them fairly means re-implementing the same
backtest three times; and taking a *gradient through* an allocation — the thing
modern, learning-based methods depend on — is simply impossible when the pieces
don't share a numerical foundation.

**jaxfolio unifies them on a single differentiable core.** Sixteen optimizers —
classical, learning-based, and graph-based — sit behind one interface,
`method(returns) → PortfolioResult`, and every constrained method is the *same*
jit-compiled projected-gradient solver with a different objective. Because the
whole pipeline (moment estimation → optimization → backtest) is JAX, it is
end-to-end differentiable and fast: you can backtest thousands of rebalances,
differentiate through an optimizer to train an allocation policy, and get exact
option Greeks for an entire chain from the same autodiff that prices it.

<br>

## Install

```bash
uv add jaxfolio            # core
uv add "jaxfolio[data]"    # + Yahoo Finance / Parquet loaders
```

## Quickstart

```python
import jaxfolio as jf
from jaxfolio.backtest import compare
from jaxfolio import viz

returns = jf.generate_returns(n_assets=10, seed=7)      # or load_yfinance / load_csv

results = compare(returns, {
    "Max Sharpe":  jf.maximum_sharpe,
    "HRP":         jf.hierarchical_risk_parity,
    "Risk Parity": jf.risk_parity,
    "1/N":         jf.equal_weight,
})

viz.save(viz.dashboard(results, returns), "dashboard.png")
```

<p align="center">
  <img src="docs/dashboard.png" alt="jaxfolio dashboard" width="880">
</p>

## Capabilities

|  |  |
|---|---|
| **Traditional** | min-variance · mean-variance · max-Sharpe · max-diversification · risk parity (ERC) · Kelly · min-CVaR · Black–Litterman |
| **Learning** | differentiable MLP Sharpe policy · online exponentiated-gradient |
| **Graph** | hierarchical risk parity (HRP) · HERC · MST centrality |
| **LLM (local)** | LLM→Black-Litterman views · news-sentiment tilt · multi-agent debate — all on a local Ollama model, no API keys |
| **Options** | Black-Scholes pricing · Greeks via autodiff · implied vol · 10+ multi-leg strategies · collar / covered-call overlays |
| **Extensible** | register your own strategy · a `toolkit` of reusable building blocks · works everywhere the built-ins do |
| **Backtest** | walk-forward engine · costs & turnover · Sharpe / Sortino / Calmar / VaR / CVaR / drawdown |
| **Data** | synthetic GBM · CSV · Parquet · Yahoo Finance · option chains |

## Options

```python
from jaxfolio.options import collar
from jaxfolio import viz

strat = collar(spot=100, put_strike=95, call_strike=110, expiry=0.25, vol=0.22)
strat.greeks(spot=100, vol=0.22)                 # net delta / gamma / vega / theta / rho
viz.save(viz.plot_payoff(strat, spot=100), "collar.png")
```

## LLM strategies (local models)

State-of-the-art LLM-driven allocation, running entirely on a **local** model via
[Ollama](https://ollama.com) — no API keys, no data leaving your machine. Each
strategy elicits per-asset **views** from the model and routes them through
Black-Litterman, so they inherit the equilibrium prior and constraints.

```bash
uv add "jaxfolio[llm]"        # adds the local-model client
ollama serve && ollama pull llama3.1
```

```python
import jaxfolio as jf
from jaxfolio.llm import OllamaClient

client = OllamaClient("llama3.1")          # any local model: mistral, qwen2.5, gemma…
returns = jf.generate_returns(n_assets=8, seed=7)

# 1 — LLM-enhanced Black-Litterman (ICLR 2025): sampled views, confidence from variance.
bl = jf.llm_black_litterman(returns, client=client, samples=5)

# 2 — News-sentiment tilt from a local model.
news = {"AAPL": "record revenue, raised guidance", "TSLA": "recall concerns"}
sent = jf.llm_sentiment_portfolio(returns, news, client=client)

# 3 — Multi-agent debate (bull / bear / risk agents negotiate the views).
agents = jf.llm_agent_portfolio(returns, client=client)

print(bl.metadata["llm_views"], bl.metadata["llm_confidence"])
```

No model installed? Every strategy accepts an injected client, so a `FakeLLM`
runs the whole flow offline (this is how the tests and
`examples/04_llm_strategies.py` work). References:
[LLM-BLM (ICLR 2025)](https://github.com/youngandbin/LLM-BLM) ·
[AlphaAgents](https://arxiv.org/abs/2508.11152) ·
[HARLF](https://arxiv.org/abs/2507.18560).

## Custom strategies

Write your own strategy and it works everywhere the built-ins do — backtester,
`compare()`, and the plots. Register it by name, or hand the shared solver a JAX
objective via the `toolkit`.

```python
import numpy as np, jax.numpy as jnp
import jaxfolio as jf
from jaxfolio.custom import custom_strategy, CustomStrategy

# Mode 1 — return weights directly (a momentum tilt).
momentum = custom_strategy(
    "momentum",
    lambda r: np.clip(((1 + r).prod() - 1).to_numpy(), 0, None),
    register=True,
)

# Mode 2 — supply a JAX objective; reuse the shared projected-gradient solver.
def entropy_minvar(w, ctx):                       # ctx exposes mu, cov, returns, assets, n
    return w @ ctx.cov @ w - 0.002 * -jnp.sum(w * jnp.log(w + 1e-9))

strat = CustomStrategy.from_objective("entropy_minvar", entropy_minvar, register=True)

jf.list_strategies(custom_only=True)              # ['entropy_minvar', 'momentum']
jf.get_strategy("momentum")(returns)              # a full PortfolioResult
```

The `jaxfolio.toolkit` module exposes the reusable building blocks —
`moments`, `make_projection`, `solve_projected_gradient`, the projections, and
`finalize_result` — so custom strategies are written the same idiomatic way as
the built-ins.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

<br>

<p align="center"><sub>MIT © jaxfolio contributors</sub></p>
