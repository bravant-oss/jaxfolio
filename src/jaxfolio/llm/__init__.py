"""Local-LLM portfolio strategies.

.. warning::

   **Experimental.** The LLM strategies in this package are research prototypes.
   Their outputs are non-deterministic and are **not** investment advice or
   investment-quality signals; their API may change without notice. Each strategy
   emits an :class:`ExperimentalWarning` on first use. See ``DISCLAIMER.md``.

All strategies run against a **local** model (Ollama by default) and never call a
cloud API. Import the strategies directly, or pass a :class:`FakeLLM` for offline
use:

>>> from jaxfolio.llm import llm_black_litterman, OllamaClient, FakeLLM
>>> result = llm_black_litterman(returns, client=OllamaClient("llama3.1"))
"""

from jaxfolio.llm._experimental import ExperimentalWarning
from jaxfolio.llm.agents import DebateResult, debate
from jaxfolio.llm.client import (
    FakeLLM,
    LLMClient,
    OllamaClient,
    default_client,
    parse_json,
)
from jaxfolio.llm.sentiment import llm_sentiment, sentiment_to_views
from jaxfolio.llm.strategies import (
    llm_agent_portfolio,
    llm_black_litterman,
    llm_sentiment_portfolio,
)
from jaxfolio.llm.views import ViewSet, llm_views

__all__ = [
    # experimental marker
    "ExperimentalWarning",
    # clients
    "LLMClient",
    "OllamaClient",
    "FakeLLM",
    "default_client",
    "parse_json",
    # building blocks
    "llm_views",
    "ViewSet",
    "llm_sentiment",
    "sentiment_to_views",
    "debate",
    "DebateResult",
    # strategies
    "llm_black_litterman",
    "llm_sentiment_portfolio",
    "llm_agent_portfolio",
]
