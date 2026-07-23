"""Tests for the deprecation mechanism (jaxfolio._deprecation).

These validate the policy machinery documented in docs/reference/stability.md:
a consistent DeprecationWarning message, and a decorator that preserves the
wrapped function's identity.
"""

from __future__ import annotations

import warnings

import pytest

from jaxfolio._deprecation import deprecated, warn_deprecated


def test_warn_deprecated_emits_message():
    with pytest.warns(DeprecationWarning) as record:
        warn_deprecated("old_thing", removal="0.3.0", replacement="new_thing")
    msg = str(record[0].message)
    assert "old_thing is deprecated" in msg
    assert "0.3.0" in msg
    assert "new_thing" in msg


def test_deprecated_decorator_warns_and_calls():
    @deprecated(removal="0.3.0", replacement="new_fn")
    def old_fn(x):
        """Original docstring."""
        return x * 2

    with pytest.warns(DeprecationWarning, match="old_fn"):
        assert old_fn(21) == 42

    # Wrapping preserves identity/metadata and flags the deprecation.
    assert old_fn.__name__ == "old_fn"
    assert old_fn.__doc__ == "Original docstring."
    assert getattr(old_fn, "__deprecated__", False) is True


def test_no_warning_when_not_called():
    """Decorating must not emit a warning until the wrapped function is called."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here would fail the test

        @deprecated(removal="0.3.0")
        def f():
            return 1

        # Definition alone raised nothing; calling it (below) is what warns.
    with pytest.warns(DeprecationWarning):
        f()
