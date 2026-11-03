"""Trusted pickle persistence with strict environment checks.

Pickle can execute arbitrary code while loading. :func:`load` therefore
requires an explicit ``trusted=True``; the environment check happens after
deserialisation and is a compatibility check, not a security boundary.
Saved models contain the training table, identifiers included.
"""

from __future__ import annotations

import os
import pickle
import platform
import tempfile
from pathlib import Path
from typing import Any

from ._environment import dependency_versions
from ._version import __version__
from .optional import optional_version
from .results import ExperimentResult
from .search import AutoCluster

__all__ = ["FORMAT_VERSION", "load", "save"]

FORMAT_VERSION: int = 2
_PICKLE_PROTOCOL: int = 5
_CHECKED_KEYS: tuple[str, ...] = ("numpy", "pandas", "scipy", "sklearn")

PathLike = str | os.PathLike[str]


def _python_series() -> tuple[str, str]:
    major, minor, _ = platform.python_version_tuple()
    return major, minor


def save(model: AutoCluster | ExperimentResult, path: PathLike) -> None:
    """Serialise a fitted AutoCluster or an ExperimentResult atomically.

    The file is written to a temporary sibling and then renamed, so an
    interrupted save never leaves a truncated file at ``path``.

    Raises
    ------
    TypeError
        If ``model`` is neither a fitted AutoCluster nor ExperimentResult.
    ValueError
        If an AutoCluster has not been fitted.
    """
    if not isinstance(model, (AutoCluster, ExperimentResult)):
        raise TypeError("Expected fitted AutoCluster or ExperimentResult.")
    if isinstance(model, AutoCluster) and not hasattr(model, "result_"):
        raise ValueError("Fit the model before saving.")
    versions = dependency_versions()
    result = model.result_ if isinstance(model, AutoCluster) else model
    uses_fuzzy = "fcm" in result.fitted_algorithms
    envelope: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "package_version": __version__,
        "python": _python_series(),
        **{key: versions[key] for key in _CHECKED_KEYS},
        "fuzzy_version": optional_version("fuzzy") if uses_fuzzy else None,
        "model": model,
    }
    target = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent or "."
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(envelope, stream, protocol=_PICKLE_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load(
    path: PathLike, *, trusted: bool = False
) -> AutoCluster | ExperimentResult:
    """Load a model saved by :func:`save` in an identical environment.

    Parameters
    ----------
    path : str or os.PathLike
        File written by :func:`save`.
    trusted : bool
        Must be ``True``; only load files you created yourself.

    Raises
    ------
    ValueError
        If ``trusted`` is false, the file is not a clustelia envelope, or
        the saved environment differs from the current one.
    """
    if not trusted:
        raise ValueError(
            "Pickle can execute code. Only load your own trusted files with "
            "trusted=True."
        )
    with open(path, "rb") as stream:
        envelope = pickle.load(stream)  # noqa: S301 - explicit trust
    if not isinstance(envelope, dict) or "model" not in envelope:
        raise ValueError("The file is not a clustelia model envelope.")
    _check_environment(envelope)
    model = envelope["model"]
    if not isinstance(model, (AutoCluster, ExperimentResult)):
        raise ValueError("Unsupported object inside the Clustelia envelope.")
    return model


def _check_environment(envelope: dict[str, Any]) -> None:
    versions = dependency_versions()
    expected: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "package_version": __version__,
        "python": _python_series(),
        **{key: versions[key] for key in _CHECKED_KEYS},
    }
    mismatched = [
        key for key, value in expected.items()
        if _normalise(envelope.get(key)) != _normalise(value)
    ]
    if mismatched:
        raise ValueError(
            "Incompatible saved environment; no cross-version migration is "
            f"supported (mismatch: {', '.join(mismatched)})."
        )
    saved_fuzzy = envelope.get("fuzzy_version")
    if saved_fuzzy is not None and saved_fuzzy != optional_version("fuzzy"):
        raise ValueError(
            "Incompatible fuzzy backend; install the saved scikit-fuzzy "
            "version."
        )


def _normalise(value: Any) -> Any:
    """Compare tuples and lists of versions by content."""
    return tuple(value) if isinstance(value, list) else value
