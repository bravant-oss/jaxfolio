# Disclaimer

jaxfolio is **research and educational software** for portfolio optimization and
options analytics. It is provided under the [MIT License](LICENSE) **as is,
without warranty of any kind**.

## Not investment advice

Nothing produced by this library — allocations, weights, signals, option prices,
Greeks, backtest results, or any other output — constitutes financial,
investment, legal, or tax advice, or a recommendation to buy or sell any
security or financial instrument. Outputs are the result of mathematical models
applied to the data and assumptions **you** provide, and models are simplified
representations of reality.

## Experimental features

Some components are explicitly **experimental** and are research prototypes, not
production-grade signals:

- **LLM-driven strategies** (`llm_black_litterman`, `llm_sentiment_portfolio`,
  `llm_agent_portfolio`) elicit views from a language model. LLM outputs are
  non-deterministic, can be confidently wrong, and **must not** be interpreted as
  investment-quality signals. They emit an `ExperimentalWarning` on first use.
- The **options execution framework** simulates fills against a *modeled*
  volatility surface. It is a research/backtesting tool, **not** a live broker,
  order-management system, or connection to any exchange or venue. Simulated
  fills, costs, and P&L will differ from live trading.

## Your responsibility

Backtested or simulated performance is **not** indicative of future results and
is subject to survivorship bias, look-ahead bias, overfitting, and unrealistic
execution assumptions. Before relying on any output for real capital, you are
responsible for independently validating the models, data, and assumptions, and
for complying with all applicable laws and regulations in your jurisdiction.

The authors and contributors accept **no liability** for any loss or damage
arising from use of this software.
