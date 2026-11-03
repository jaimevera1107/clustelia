"""Input validation and schema resolution.

The validator returns an owned copy of the input; callers never mutate the
user's object. Sparse inputs are rejected rather than densified silently.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Schema
from .exceptions import CapabilityError

__all__ = [
    "HIGH_CARDINALITY_FRACTION",
    "MIN_TRAINING_ROWS",
    "ValidatedTable",
    "validate_table",
]

MIN_TRAINING_ROWS: int = 4
"""Smallest number of rows accepted for fitting."""

HIGH_CARDINALITY_FRACTION: float = 0.5
"""Categorical columns with more unique levels than this fraction of rows
are reported as high-cardinality (diagnostic only)."""


class ValidatedTable(NamedTuple):
    """Owned, validated input and the facts established about it."""

    frame: pd.DataFrame
    schema: Schema
    diagnostics: dict[str, Any]


def validate_table(
    X: Any,
    schema: Schema | None = None,
    expected_columns: Any = None,
    min_rows: int = MIN_TRAINING_ROWS,
    allow_missing: bool = False,
) -> ValidatedTable:
    """Validate a two-dimensional table and resolve its schema.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Input table. NumPy inputs use row positions as identities.
    schema : Schema or None
        Explicit column roles. When ``None`` numeric dtypes become numeric
        features and every other column becomes categorical; ordinal and
        binary semantics are never inferred.
    expected_columns : sequence or None
        Exact column names and order required (used for assignment).
    min_rows : int
        Minimum number of rows.
    allow_missing : bool
        Accept missing feature values (Gower pairwise deletion). Each row
        must still have at least one observed feature.

    Returns
    -------
    ValidatedTable
        Owned frame, resolved schema and diagnostics.

    Raises
    ------
    CapabilityError
        If ``X`` is a sparse matrix.
    ValueError
        If the table violates any input contract.
    """
    if sparse.issparse(X):
        raise CapabilityError("Sparse matrices are not supported.")
    frame = _owned_frame(X)
    _check_shape_and_identity(frame, min_rows, expected_columns)
    inferred = schema is None
    resolved = _infer_schema(frame) if schema is None else schema
    _check_schema_coverage(frame, resolved)
    _check_identifiers(frame, resolved)
    _check_features(frame, resolved, allow_missing)
    diagnostics = _diagnose(frame, resolved, inferred, X)
    return ValidatedTable(frame, resolved, diagnostics)


def _owned_frame(X: Any) -> pd.DataFrame:
    """Return a deep copy of ``X`` as a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=True)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    return pd.DataFrame(values.copy())


def _check_shape_and_identity(
    frame: pd.DataFrame, min_rows: int, expected_columns: Any
) -> None:
    if len(frame) < min_rows or frame.shape[1] == 0:
        raise ValueError("Insufficient rows or no features.")
    if (
        not frame.columns.is_unique
        or not frame.index.is_unique
        or frame.index.hasnans
    ):
        raise ValueError(
            "Columns and row identifiers must be unique; row identifiers "
            "cannot be missing."
        )
    if expected_columns is not None and list(frame.columns) != list(
        expected_columns
    ):
        raise ValueError(
            "Prediction columns and their order must exactly match training."
        )


def _infer_schema(frame: pd.DataFrame) -> Schema:
    numeric = tuple(frame.select_dtypes(include=[np.number]).columns)
    categorical = tuple(c for c in frame.columns if c not in numeric)
    return Schema(numeric=numeric, categorical=categorical)


def _check_schema_coverage(frame: pd.DataFrame, schema: Schema) -> None:
    if set(schema.columns) != set(frame.columns):
        raise ValueError(
            "Schema must declare every input column exactly once."
        )
    if not schema.features:
        raise ValueError(
            "At least one feature is required after excluding identifiers."
        )


def _check_identifiers(frame: pd.DataFrame, schema: Schema) -> None:
    if not schema.identifiers:
        return
    identifiers = frame.loc[:, list(schema.identifiers)]
    if identifiers.isna().any().any() or identifiers.duplicated().any():
        raise ValueError(
            "Declared entity identifiers must be complete and unique."
        )


def _check_features(
    frame: pd.DataFrame, schema: Schema, allow_missing: bool = False
) -> None:
    features = frame.loc[:, list(schema.features)]
    missing = features.isna()
    if missing.any().any():
        if not allow_missing:
            raise ValueError(
                "Missing features: the missing-value policy is 'error'; "
                "impute explicitly upstream or use "
                "GeometrySpec(kind='gower', scaling='none', "
                "missing='pairwise')."
            )
        if missing.all(axis=1).any():
            raise ValueError(
                "Every row needs at least one observed feature under "
                "pairwise deletion."
            )
    for column in schema.numeric:
        dtype = frame[column].dtype
        if not pd.api.types.is_numeric_dtype(
            dtype
        ) or pd.api.types.is_complex_dtype(dtype):
            raise ValueError(
                f"Numeric column {column!r} must have a real numeric dtype."
            )
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values[~np.isnan(values)]).all():
            raise ValueError(f"Non-finite values in {column!r}.")
    for column, levels in schema.ordinal:
        observed = frame[column].dropna()
        if not observed.isin(levels).all():
            raise ValueError(
                f"Ordinal column {column!r} contains values outside its "
                "declared levels."
            )
    for column in schema.binary:
        if not _is_binary(frame[column].dropna()):
            raise ValueError(
                f"Binary column {column!r} must contain only 0/1 or "
                "False/True."
            )


def _is_binary(values: pd.Series) -> bool:
    """Whether a column holds only 0/1 (or boolean) values."""
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if not pd.api.types.is_numeric_dtype(dtype):
        return False
    return bool(values.isin((0, 1)).all())


def _diagnose(
    frame: pd.DataFrame, schema: Schema, inferred: bool, original: Any
) -> dict[str, Any]:
    features = frame.loc[:, list(schema.features)]
    limit = len(frame) * HIGH_CARDINALITY_FRACTION
    if schema.identifiers:
        row_identity = "declared_columns"
    elif isinstance(original, pd.DataFrame):
        row_identity = "dataframe_index"
    else:
        row_identity = "row_position"
    return {
        "schema_inferred": inferred,
        "excluded_identifiers": list(schema.identifiers),
        "constant_columns": [
            c for c in features if features[c].nunique() <= 1
        ],
        "high_cardinality_categoricals": [
            c for c in schema.categorical if frame[c].nunique() > limit
        ],
        "row_identity": row_identity,
    }
