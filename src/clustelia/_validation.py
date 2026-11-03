"""Scalar validators shared by the configuration objects.

Validators normalise accepted values to built-in Python types (for example
``numpy.int64`` to ``int``). Normalisation keeps configuration hashes and
candidate identifiers stable regardless of the numeric type supplied.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

__all__ = [
    "is_integer",
    "require_finite",
    "require_int",
    "require_nonnegative",
    "require_open_unit",
    "require_positive",
    "require_unit_interval",
]


def is_integer(value: object) -> bool:
    """Return whether ``value`` is integral and not a boolean.

    Parameters
    ----------
    value : object
        Value to inspect.

    Returns
    -------
    bool
        ``True`` for ``int`` and NumPy integer scalars, ``False`` for
        booleans and every other type.
    """
    return isinstance(value, Integral) and not isinstance(value, bool)


def require_int(name: str, value: object, minimum: int) -> int:
    """Validate an integer bounded from below.

    Parameters
    ----------
    name : str
        Parameter name used in the error message.
    value : object
        Value to validate.
    minimum : int
        Smallest accepted value.

    Returns
    -------
    int
        The value as a built-in ``int``.

    Raises
    ------
    ValueError
        If the value is not an integer or is below ``minimum``.
    """
    message = f"{name} must be an integer >= {minimum}."
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(message)
    number = int(value)
    if number < minimum:
        raise ValueError(message)
    return number


def require_finite(name: str, value: object) -> float:
    """Validate a finite real number and return it as ``float``.

    Parameters
    ----------
    name : str
        Parameter name used in the error message.
    value : object
        Value to validate.

    Returns
    -------
    float
        The value as a built-in ``float``.

    Raises
    ------
    ValueError
        If the value is not a finite real number.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def require_positive(name: str, value: object) -> float:
    """Validate a finite, strictly positive real number."""
    number = require_finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def require_nonnegative(name: str, value: object) -> float:
    """Validate a finite, nonnegative real number."""
    number = require_finite(name, value)
    if number < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


def require_unit_interval(name: str, value: object) -> float:
    """Validate a finite real number in the closed interval [0, 1]."""
    number = require_finite(name, value)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite in [0, 1].")
    return number


def require_open_unit(name: str, value: object) -> float:
    """Validate a finite real number in the open interval (0, 1)."""
    number = require_finite(name, value)
    if not 0 < number < 1:
        raise ValueError(f"{name} must be in (0, 1).")
    return number
