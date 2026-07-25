# LLM strategies

!!! warning "Experimental — not investment advice"
    These strategies are **experimental research prototypes**. LLM outputs are
    non-deterministic and can be confidently wrong; the resulting allocations are
    **not** investment advice or investment-quality signals, and the API may
    change without notice. Each strategy emits an `ExperimentalWarning` on first
    use. See the [disclaimer](https://github.com/bravant-oss/jaxfolio/blob/main/DISCLAIMER.md)
    and the [stability policy](../reference/stability.md#experimental-features).

jaxfolio ships three LLM-driven allocation strategies that run entirely on a
**local** model via [Ollama](https://ollama.com) — no API keys, and no data
leaves your machine. Each elicits per-asset **views** from the model and routes
them through [`black_litterman`](optimizers.md#black_litterman), so they inherit
its equilibrium prior, constraints, and diagnostics for free.

!!! info "The design in one line"
    An LLM is only used to produce *views* and a *confidence*; the actual
    allocation is still Black–Litterman. This keeps the output well-behaved and
    grounded in market equilibrium, whatever the model says.

## The client abstraction

Strategies never talk to a cloud API. They depend on a small
[`LLMClient`](../reference/llm.md#jaxfolio.llm.client.LLMClient) protocol with
`complete` and `sample` methods, and accept an **injected** client:

- [`OllamaClient`](../reference/llm.md#jaxfolio.llm.client.OllamaClient) — drives a
  local Ollama server (`http://localhost:11434`). Because Ollama also exposes an
  OpenAI-compatible endpoint, the same host serves llama.cpp / LM Studio / vLLM.
- [`FakeLLM`](../reference/llm.md#jaxfolio.llm.client.FakeLLM) — a deterministic,
  offline stand-in that returns canned responses with no network. This is how the
  tests and examples run without a model installed.

```python
from jaxfolio.llm import OllamaClient, FakeLLM

client = OllamaClient("llama3.1")          # any local model: mistral, qwen2.5, gemma3…
# or, fully offline:
client = FakeLLM(['{"ASSET_00": 0.01, "ASSET_01": -0.005}'])
```

Get a model running first:

```bash
uv add "jaxfolio[llm]"
ollama serve && ollama pull llama3.1
```

## 1 — LLM-enhanced Black–Litterman

[`llm_black_litterman`](../reference/llm.md#jaxfolio.llm.strategies.llm_black_litterman)
implements the core of the ICLR-2025 method *Integrating LLM-Generated Views into
Mean–Variance Optimization*. It shows the model each asset's recent return
statistics, **samples it `k` times**, parses per-asset expected-return views, and
turns the *dispersion across samples* into a confidence — so uncertain views are
automatically down-weighted before entering Black–Litterman.

```python
import jaxfolio as jf
from jaxfolio.llm import OllamaClient

returns = jf.generate_returns(n_assets=8, seed=7)
client = OllamaClient("llama3.1")

bl = jf.llm_black_litterman(returns, client=client, samples=5)

bl.metadata["llm_views"]        # {asset: mean sampled view}
bl.metadata["llm_confidence"]   # (0, 1] — high when samples agreed
bl.metadata["llm_view_std"]     # per-asset dispersion
```

Confidence is calibrated from agreement across samples: tight agreement (low std
relative to the spread of views) maps to high confidence.

## 2 — News-sentiment tilt

[`llm_sentiment_portfolio`](../reference/llm.md#jaxfolio.llm.strategies.llm_sentiment_portfolio)
scores per-asset sentiment in \([-1, 1]\) from headlines/notes, converts each
score to a small return tilt (`score × strength`), and combines it with the
market equilibrium through Black–Litterman.

```python
news = {
    "AAPL": "record revenue, raised guidance",
    "TSLA": "recall concerns",
}
sent = jf.llm_sentiment_portfolio(returns, news, client=client, strength=0.03)

sent.metadata["sentiment_scores"]   # {asset: score in [-1, 1]}
sent.metadata["sentiment_views"]    # {asset: return tilt}
```

Each asset's `news` value may be a single string or a list of headlines.

## 3 — Multi-agent debate

[`llm_agent_portfolio`](../reference/llm.md#jaxfolio.llm.strategies.llm_agent_portfolio)
runs an AlphaAgents / HARLF-style debate. Role-specialized agents — a **bull**, a
**bear**, and a **risk manager** by default — each argue per-asset views from the
same data. A moderator averages them per asset, with confidence reflecting
cross-agent agreement, and Black–Litterman turns the consensus into weights.

```python
agent = jf.llm_agent_portfolio(returns, client=client)

agent.metadata["agent_views"]           # each role's individual stance
agent.metadata["consensus_views"]       # the aggregated views
agent.metadata["consensus_confidence"]
```

The value over a single prompt is *structured disagreement*: the bull looks for
upside, the bear for downside, the risk agent penalizes volatility, and the
aggregation reflects where they converge.

## Running offline

Every strategy accepts an injected client, so the entire flow runs with no model
installed by passing a `FakeLLM`. A callable responder makes the offline data
realistic enough to exercise the confidence calibration:

```python
import json, numpy as np
from jaxfolio.llm import FakeLLM

assets = list(returns.columns)

def responder(prompt: str, i: int) -> str:
    rng = np.random.default_rng(i + 1)
    return json.dumps({a: round(float(rng.normal(0.01, 0.006)), 4) for a in assets})

bl = jf.llm_black_litterman(returns, client=FakeLLM(responder), samples=5)
```

## Using inside `compare`

Because these strategies take a client argument, bind it with `functools.partial`
before handing them to the [backtester](backtesting.md):

```python
from functools import partial
from jaxfolio.backtest import compare

compare(returns, {
    "LLM-BL": partial(jf.llm_black_litterman, client=client, samples=5),
    "1/N":    jf.equal_weight,
})
```

## References

- **LLM-BL (ICLR 2025)** — [Integrating LLM-Generated Views into Mean-Variance Optimization](https://github.com/youngandbin/LLM-BLM)
- **AlphaAgents** — [arXiv:2508.11152](https://arxiv.org/abs/2508.11152)
- **HARLF** — [arXiv:2507.18560](https://arxiv.org/abs/2507.18560)
