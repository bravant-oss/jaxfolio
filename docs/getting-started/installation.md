# Installation

jaxfolio targets **Python 3.11+** and installs a small, well-scoped dependency
set. Optional extras add data loaders and the local-LLM client only when you
need them, so the core install stays lean.

## Core install

=== "uv"

    ```bash
    uv add jaxfolio
    ```

=== "pip"

    ```bash
    pip install jaxfolio
    ```

The core install pulls in `jax`, `jaxlib`, `optax`, `numpy`, `pandas`, `scipy`,
and `matplotlib` — everything needed for the optimizers, the backtester, the
options toolkit, and the visualizations.

## Optional extras

jaxfolio splits network- and integration-heavy dependencies into extras. Install
only what you use.

| Extra | Installs | Enables |
|---|---|---|
| `data` | `yfinance`, `pyarrow` | [`load_yfinance`](../reference/data.md), [`load_parquet`](../reference/data.md), [`load_option_chain`](../reference/options.md) |
| `llm`  | `requests` | the [`OllamaClient`](../guide/llm.md) local-model client |
| `dev`  | `pytest`, `pytest-cov`, `ruff` | the test suite and linters |

=== "uv"

    ```bash
    uv add "jaxfolio[data]"          # + Yahoo Finance / Parquet loaders
    uv add "jaxfolio[llm]"           # + local-model client
    uv add "jaxfolio[data,llm]"      # both
    ```

=== "pip"

    ```bash
    pip install "jaxfolio[data]"
    pip install "jaxfolio[llm]"
    pip install "jaxfolio[data,llm]"
    ```

!!! note "Extras are guarded"
    Importing jaxfolio never requires an extra. The loaders and the LLM client
    raise a clear, actionable `ImportError` only if you call them without the
    corresponding extra installed — so the core package always imports cleanly.

## Verify the installation

```python
import jaxfolio as jf

print(jf.__version__)

returns = jf.generate_returns(n_assets=6, seed=0)   # offline synthetic data
result = jf.maximum_sharpe(returns)
print(result)                                       # PortfolioResult(method='Maximum Sharpe', ...)
```

If that prints a `PortfolioResult`, you are ready to go — no network access is
required, because [`generate_returns`](../reference/data.md) produces reproducible
synthetic data locally.

## Local LLM prerequisites (optional)

The [LLM strategies](../guide/llm.md) drive a **local** model via
[Ollama](https://ollama.com) — no API keys, and no data leaves your machine.
After installing the `llm` extra:

```bash
# 1. install Ollama:  https://ollama.com
ollama serve             # start the local server
ollama pull llama3.1     # or mistral, qwen2.5, gemma3, ...
```

Every LLM strategy also accepts an injected client, so you can run the entire
flow offline with the built-in `FakeLLM` — this is how the tests and
[`examples/04_llm_strategies.py`](https://github.com/bravant-oss/jaxfolio/blob/main/examples/04_llm_strategies.py)
work.

## Development setup

Clone the repository and sync all extras:

```bash
git clone https://github.com/bravant-oss/jaxfolio
cd jaxfolio

uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

To build these docs locally:

```bash
uv sync --extra docs
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
uv run mkdocs build          # render the static site into ./site
```
