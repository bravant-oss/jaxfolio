"""Deprecation utilities.

A single, consistent mechanism for the deprecation policy documented in
``docs/reference/stability.md``: emit a ``DeprecationWarning`` for at least one
minor release before a public name is removed or its behavior changed.

The warnings are attributed to the caller (``stacklevel`` is adjusted) and carry
the ``jaxfolio`` module in their category chain so tests can turn *our*
deprecations into errors while still ignoring third-party ones (see the
``filterwarnings`` config in ``pyproject.toml``).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


def _format(name: str, *, removal: str | None, replacement: str | None, reason: str | None) -> str:
    msg = f"{name} is deprecated"
    if removal:
        msg += f" and will be removed in {removal}"
    msg += "."
    if replacement:
        msg += f" Use {replacement} instead."
    if reason:
        msg += f" {reason}"
    return msg


def warn_deprecated(
    name: str,
    *,
    removal: str | None = None,
    replacement: str | None = None,
    reason: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a standardized ``DeprecationWarning`` for a deprecated code path.

    Parameters
    ----------
    name:
        The deprecated object or behavior, e.g. ``"jaxfolio.old_fn()"``.
    removal:
        Version in which the name is scheduled for removal, e.g. ``"0.3.0"``.
    replacement:
        The recommended replacement, if any.
    reason:
        Optional extra context appended to the message.
    stacklevel:
        Passed to :func:`warnings.warn`; the default points the warning at the
        immediate caller.
    """
    warnings.warn(
        _format(name, removal=removal, replacement=replacement, reason=reason),
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(
    *,
    removal: str | None = None,
    replacement: str | None = None,
    reason: str | None = None,
) -> Callable[[F], F]:
    """Decorate a function or method as deprecated.

    Wrapping preserves the original signature and docstring; calling the wrapped
    object emits a :class:`DeprecationWarning` via :func:`warn_deprecated`.

    Examples
    --------
    >>> @deprecated(removal="0.3.0", replacement="new_fn")
    ... def old_fn(x):
    ...     return x
    """

    def decorate(func: F) -> F:
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            warn_deprecated(
                f"{qualname}()",
                removal=removal,
                replacement=replacement,
                reason=reason,
                stacklevel=3,
            )
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
