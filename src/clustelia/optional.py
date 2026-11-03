"""Lazy, version-pinned optional dependencies.

Importing clustelia never imports an optional backend. A backend is usable
only when the exact pinned version is installed, because DBCV and FCM
results are part of the documented methodology.
"""

from __future__ import annotations

from importlib import import_module, metadata
from types import ModuleType
from typing import NamedTuple

from .exceptions import CapabilityError

__all__ = ["OPTIONALS", "OptionalBackend", "optional_version",
           "require_optional"]


class OptionalBackend(NamedTuple):
    """Distribution name, import name and pinned version of a backend."""

    distribution: str
    module: str
    required_version: str


OPTIONALS: dict[str, OptionalBackend] = {
    "density": OptionalBackend("hdbscan", "hdbscan", "0.8.44"),
    "fuzzy": OptionalBackend("scikit-fuzzy", "skfuzzy", "0.5.0"),
}


def optional_version(feature: str) -> str | None:
    """Return the installed version if it matches the pin, else ``None``.

    Parameters
    ----------
    feature : {"density", "fuzzy"}
        Optional feature name.

    Raises
    ------
    KeyError
        If ``feature`` is not a known optional feature.
    """
    backend = OPTIONALS[feature]
    try:
        installed = metadata.version(backend.distribution)
    except metadata.PackageNotFoundError:
        return None
    return installed if installed == backend.required_version else None


def require_optional(feature: str) -> ModuleType:
    """Import and return the backend module of ``feature``.

    Raises
    ------
    CapabilityError
        If the pinned backend version is not installed.
    """
    backend = OPTIONALS[feature]
    if optional_version(feature) is None:
        raise CapabilityError(
            f"Install clustelia[{feature}] "
            f"({backend.distribution}=={backend.required_version}) for this "
            "capability."
        )
    return import_module(backend.module)
