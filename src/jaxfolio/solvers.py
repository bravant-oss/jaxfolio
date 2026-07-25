"""Solver selection: the built-in SPG method plus any optax optimizer.

The shared projected-gradient solver in :mod:`jaxfolio.optimizers.base` accepts a
*solver spec* rather than a hard-coded name. A spec is produced by
:func:`resolve_solver`, which understands three spellings:

* ``"spg"`` — the built-in spectral projected-gradient method (default).
* any optax optimizer **by name** — ``"adam"``, ``"adamw"``, ``"sgd"``,
  ``"rmsprop"``, ``"lion"``, ... Resolved against :mod:`optax` and then
  :mod:`optax.contrib`, so anything the installed optax ships is reachable.
* an optax **factory callable** — ``optax.adamw``, ``optax.contrib.adopt``, or a
  module-level function of your own that returns a ``GradientTransformation``
  (the escape hatch for ``optax.chain(...)`` compositions).

Hyperparameters go in ``solver_options`` (``{"weight_decay": 1e-3}``), *not* in a
pre-built transformation — see the cache note below.

Cache stability
---------------
The solver spec is a **static** argument of the jit-cached kernel, so it has to be
hashable and compare equal across separately constructed instances. Name strings
and module-level factories satisfy that: ``optax.adamw`` is the same object every
time you look it up, and :class:`SolverSpec` is a plain ``NamedTuple`` of
hashable fields. A *pre-built* transformation (``optax.adam(1e-2)``) does not: it
is a fresh tuple of closures on every call, hashed by identity, so passing one
would recompile the kernel on every single solve. :func:`resolve_solver`
therefore rejects pre-built transformations and points at the factory spelling.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp
import optax

__all__ = [
    "SolverSpec",
    "available_solvers",
    "build_optimizer",
    "resolve_solver",
]

SPG = "spg"

#: Default step size for every optax solver when ``learning_rate`` is ``None``.
DEFAULT_LEARNING_RATE = 1e-2


class SolverSpec(NamedTuple):
    """A hashable, equality-stable solver key (safe as a jit static argument).

    Attributes
    ----------
    kind:
        ``"spg"`` for the built-in spectral projected gradient, ``"optax"`` for
        everything else.
    factory:
        The optax factory to call, e.g. ``optax.adamw``. ``None`` for ``"spg"``.
    name:
        Display label used in error messages and diagnostics.
    options:
        Extra keyword arguments for the factory, normalized to a sorted tuple of
        ``(key, value)`` pairs so the spec stays hashable.
    """

    kind: str = SPG
    factory: Callable[..., Any] | None = None
    name: str = SPG
    options: tuple[tuple[str, Any], ...] = ()


# Singleton for the (overwhelmingly common) default, so every ``OptimizerConfig()``
# yields the *same* jit cache key without rebuilding a tuple each time.
_SPG_SPEC = SolverSpec()


def _normalize_options(
    options: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...]:
    """Freeze a mapping of solver hyperparameters into a sorted, hashable tuple."""
    if not options:
        return ()
    items = dict(options)
    for key, value in items.items():
        if not isinstance(key, str):
            raise TypeError(f"solver_options keys must be strings, got {key!r}")
        if key == "learning_rate":
            raise ValueError(
                "set the step size via the config's `learning_rate` field, "
                "not solver_options['learning_rate']"
            )
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"solver_options[{key!r}] must be hashable (got {type(value).__name__}); "
                "solver options become part of the jit cache key"
            ) from exc
    return tuple(sorted(items.items()))


def _lookup(name: str) -> Callable[..., Any]:
    """Find an optax factory by name, searching ``optax`` then ``optax.contrib``."""
    for module in (optax, optax.contrib):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    suggestions = difflib.get_close_matches(name, available_solvers(), n=3)
    hint = f" Did you mean {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
    raise ValueError(
        f"unknown solver {name!r}; expected 'spg' or the name of an optax optimizer."
        f"{hint} See jaxfolio.solvers.available_solvers() for the full list."
    )


def resolve_solver(
    solver: str | Callable[..., Any] | SolverSpec = SPG,
    solver_options: Mapping[str, Any] | None = None,
) -> SolverSpec:
    """Normalize a solver spelling into a :class:`SolverSpec`.

    Accepts ``"spg"``, an optax optimizer name, an optax factory callable, or an
    already-resolved :class:`SolverSpec` (the function is idempotent, so callers
    can normalize defensively). Raises :class:`ValueError` for an unknown name and
    :class:`TypeError` for an unusable object.
    """
    if isinstance(solver, SolverSpec):
        return solver

    options = _normalize_options(solver_options)

    if solver == SPG:
        if options:
            raise ValueError(
                "solver_options apply to optax solvers only; 'spg' is tuned by "
                "`learning_rate` (its initial Barzilai-Borwein step) and `tol`"
            )
        return _SPG_SPEC

    if isinstance(solver, str):
        factory = _lookup(solver)
        return SolverSpec("optax", factory, solver, options)

    # ``optax.nadam``/``nadamw`` are functools.partial objects, so gate on
    # callable() rather than inspect.isfunction().
    if callable(solver):
        name = getattr(solver, "__name__", None) or repr(solver)
        return SolverSpec("optax", solver, name, options)

    if hasattr(solver, "init") and hasattr(solver, "update"):
        raise TypeError(
            "pass the optax *factory* rather than a pre-built GradientTransformation "
            "(e.g. solver=optax.adamw with solver_options={'weight_decay': 1e-3}, not "
            "solver=optax.adamw(1e-3)). A pre-built transformation is a fresh object on "
            "every call, so it would recompile the solver kernel on every solve. For an "
            "optax.chain(...) composition, wrap it in a module-level factory function "
            "`def my_solver(learning_rate): ...` and pass that."
        )

    raise TypeError(
        f"solver must be 'spg', an optax optimizer name, or an optax factory callable; "
        f"got {type(solver).__name__}"
    )


def _reject_line_search(tx: optax.GradientTransformation, name: str) -> None:
    """Raise if ``tx`` needs per-step extra arguments the solver loop cannot supply.

    Line-search optimizers (``optax.lbfgs``, ``optax.polyak_sgd``) require
    ``value``/``grad``/``value_fn`` on every ``update`` call. Their ``update``
    signature is the same as every other transformation's, so the only reliable
    detection is to try one step. This runs once per trace, not per iteration.
    """
    probe = jnp.zeros((2,))
    try:
        tx.update(probe, tx.init(probe), probe)
    except TypeError as exc:
        raise ValueError(
            f"solver {name!r} needs extra arguments per step (value/grad/value_fn): "
            "line-search optimizers such as optax.lbfgs and optax.polyak_sgd are not "
            "supported by the projected-gradient loop. Use 'spg' (which adapts its own "
            f"step size from the local curvature) or a first-order optax optimizer. "
            f"Original error: {exc}"
        ) from exc
    except Exception:  # noqa: BLE001 - the probe is best-effort, never a gate
        # An unrelated failure (dtype quirk, shape assumption) must not reject an
        # optimizer that would work fine on the real problem.
        pass


def build_optimizer(
    spec: SolverSpec,
    learning_rate: float | None = None,
) -> optax.GradientTransformation:
    """Instantiate the optax transformation described by ``spec``.

    ``learning_rate=None`` falls back to :data:`DEFAULT_LEARNING_RATE` (``1e-2``)
    — optax factories have no default of their own. Factories that take no
    learning rate (``optax.contrib.dowg``, ``cocob``, ...) are called without one,
    and an explicitly supplied rate is then an error rather than silently ignored.
    """
    if spec.kind != "optax" or spec.factory is None:
        raise ValueError(f"cannot build an optax optimizer from solver kind {spec.kind!r}")

    kwargs: dict[str, Any] = dict(spec.options)
    try:
        params = inspect.signature(spec.factory).parameters
        takes_lr = "learning_rate" in params
    except (ValueError, TypeError):  # builtins / C-implemented callables
        takes_lr = True

    if takes_lr:
        # Always by keyword: contrib factories order their arguments differently.
        kwargs["learning_rate"] = DEFAULT_LEARNING_RATE if learning_rate is None else learning_rate
    elif learning_rate is not None:
        raise ValueError(f"solver {spec.name!r} does not take a learning_rate; leave it as None")

    try:
        tx = spec.factory(**kwargs)
    except TypeError as exc:
        given = dict(spec.options)
        raise ValueError(f"solver {spec.name!r} rejected solver_options {given}: {exc}") from exc

    if not (hasattr(tx, "init") and hasattr(tx, "update")):
        raise ValueError(
            f"solver {spec.name!r} did not return an optax GradientTransformation "
            f"(got {type(tx).__name__}); it is probably not an optimizer"
        )

    _reject_line_search(tx, spec.name)
    return tx


def available_solvers() -> tuple[str, ...]:
    """Return the solver names that can be passed as ``solver=...``.

    ``("spg",)`` followed by the sorted names of every optax and ``optax.contrib``
    factory that takes a ``learning_rate``. This is a discovery aid for humans and
    for error messages — resolution itself is more permissive, so optimizers with
    their own adaptive step (``dowg``, ``cocob``) still work even though they are
    not listed here.
    """
    names: set[str] = set()
    for module in (optax, optax.contrib):
        for name in dir(module):
            if name.startswith("_") or name.startswith("scale_by_"):
                continue
            candidate = getattr(module, name, None)
            if not callable(candidate) or inspect.isclass(candidate):
                continue
            try:
                params = inspect.signature(candidate).parameters
            except (ValueError, TypeError):
                continue
            if "learning_rate" in params:
                names.add(name)
    return (SPG, *sorted(names))
