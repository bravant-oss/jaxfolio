"""Experimental-feature warning machinery for the LLM strategies.

Kept in its own module (imported by both ``llm/__init__.py`` and
``llm/strategies.py``) to avoid an import cycle.
"""

from __future__ import annotations

import warnings


class ExperimentalWarning(UserWarning):
    """Signals that a feature is an experimental research prototype.

    Its API and behavior may change without notice, and its outputs are **not**
    investment advice or investment-quality signals.
    """


_WARNED = False

_MESSAGE = (
    "{feature} is an EXPERIMENTAL research feature: its API and behavior may change "
    "without notice, and its outputs are NOT investment advice or investment-quality "
    "signals (see jaxfolio's DISCLAIMER.md). Silence this with "
    "warnings.filterwarnings('ignore', category="
    "jaxfolio.llm.ExperimentalWarning)."
)


def warn_experimental(feature: str, *, stacklevel: int = 3) -> None:
    """Emit a one-time :class:`ExperimentalWarning` for ``feature``.

    Fires at most once per process (across all LLM strategies) so it flags the
    experimental status without spamming repeated backtest calls.
    """
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(_MESSAGE.format(feature=feature), ExperimentalWarning, stacklevel=stacklevel)


def _reset_experimental_warning() -> None:
    """Test helper: re-arm the one-time warning so it can fire again."""
    global _WARNED
    _WARNED = False
