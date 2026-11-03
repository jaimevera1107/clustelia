"""Versions of the runtime environment, recorded for reproducibility."""

from __future__ import annotations

import platform

import numpy
import pandas
import scipy
import sklearn

from .optional import optional_version

__all__ = ["dependency_versions"]


def dependency_versions() -> dict[str, str | None]:
    """Return the versions of Python and of every relevant dependency."""
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "hdbscan_validity": optional_version("density"),
        "scikit-fuzzy": optional_version("fuzzy"),
    }
