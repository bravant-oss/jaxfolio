"""Local-LLM client abstraction.

The strategies in this package never talk to a cloud API — they drive a **local**
model. :class:`OllamaClient` speaks to a local `Ollama <https://ollama.com>`_
server (``http://localhost:11434``); because Ollama also exposes an
OpenAI-compatible endpoint, the same host serves llama.cpp / LM Studio / vLLM.

Everything is built around the small :class:`LLMClient` protocol, so strategies
accept an injected client. Tests and fully-offline runs pass :class:`FakeLLM`,
which returns canned responses with no network at all.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"


@runtime_checkable
class LLMClient(Protocol):
    """The minimal interface a strategy needs from a local LLM."""

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> str:
        """Return a single completion for ``prompt``."""
        ...

    def sample(
        self, prompt: str, k: int, *, system: str | None = None, temperature: float = 0.7
    ) -> list[str]:
        """Return ``k`` independent completions (for uncertainty estimation)."""
        ...


class OllamaClient:
    """Client for a local Ollama server (native ``/api/generate`` endpoint).

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.1"``, ``"mistral"``, ``"qwen2.5"``.
        Pull it first with ``ollama pull <model>``.
    host:
        Base URL of the local server.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        options: dict | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.options = options or {}

    def _post(self, prompt: str, system: str | None, temperature: float) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OllamaClient requires the optional 'llm' extra: "
                "install with `uv sync --extra llm` or `pip install jaxfolio[llm]`."
            ) from exc

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, **self.options},
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise ConnectionError(
                f"Could not reach a local Ollama server at {self.host}. "
                "Start it and pull a model:\n"
                "  1. Install Ollama: https://ollama.com\n"
                "  2. Run the server: `ollama serve`\n"
                f"  3. Pull the model: `ollama pull {self.model}`\n"
                f"Original error: {exc}"
            ) from exc
        return resp.json().get("response", "")

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> str:
        return self._post(prompt, system, temperature)

    def sample(
        self, prompt: str, k: int, *, system: str | None = None, temperature: float = 0.7
    ) -> list[str]:
        # Ollama has no native n>1; issue k independent requests.
        return [self._post(prompt, system, temperature) for _ in range(k)]


class FakeLLM:
    """A deterministic, offline stand-in for a local LLM.

    Pass either a fixed list of responses (cycled), or a callable
    ``responder(prompt, index) -> str`` for prompt-dependent behavior. Used in the
    test suite and for running the examples without a model installed.
    """

    def __init__(self, responses: Sequence[str] | Callable[[str, int], str]):
        self._responses = responses
        self._calls = 0

    def _next(self, prompt: str) -> str:
        i = self._calls
        self._calls += 1
        if callable(self._responses):
            return self._responses(prompt, i)
        seq = list(self._responses)
        return seq[i % len(seq)] if seq else ""

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> str:
        return self._next(prompt)

    def sample(
        self, prompt: str, k: int, *, system: str | None = None, temperature: float = 0.7
    ) -> list[str]:
        return [self._next(prompt) for _ in range(k)]

    @property
    def call_count(self) -> int:
        return self._calls


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response, tolerating prose.

    Handles ```json fenced blocks and leading/trailing commentary. Returns an
    empty dict if nothing parseable is found (callers treat that as "no view").
    """
    if not text:
        return {}

    # Try candidates in order of preference; the first that parses to a dict wins.
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    # Fall back to every brace-balanced object in the text (skipping braces inside
    # strings so a `}` in a value doesn't close early), so a failed fence or an
    # invalid earlier object doesn't hide a valid one later in the response.
    candidates.extend(_json_objects(text))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _json_objects(text: str) -> list[str]:
    """Return every top-level brace-balanced ``{...}`` substring, ignoring braces in strings."""
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                objects.append(text[start : i + 1])
    return objects


def default_client(model: str = DEFAULT_MODEL, **kwargs) -> OllamaClient:
    """Construct the default local client (Ollama)."""
    return OllamaClient(model, **kwargs)
