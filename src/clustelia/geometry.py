"""Fitted reference geometry, independent of each candidate's fit space.

:class:`ReferenceGeometry` selects the nonconstant features and delegates
the metric itself to a small strategy object (composition): euclidean with
optional scaling, or weighted Gower for numeric, nominal, ordinal and
asymmetric binary variables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol, Self, TypeAlias

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import GeometrySpec, Schema
from .exceptions import CapabilityError, MemoryBudgetError

__all__ = [
    "BYTES_PER_FLOAT64",
    "DEFAULT_PAIRWISE_MATRICES",
    "EuclideanMetric",
    "GowerMetric",
    "ReferenceGeometry",
    "estimate_pairwise_mib",
    "guard_pairwise",
]

BYTES_PER_FLOAT64: int = 8
DEFAULT_PAIRWISE_MATRICES: int = 3
"""Dense ``n x n`` float64 matrices assumed alive at the same time."""

_MIB: int = 1024**2

GeometryValues: TypeAlias = np.ndarray | pd.DataFrame


def estimate_pairwise_mib(
    n_rows: int, matrices: int = DEFAULT_PAIRWISE_MATRICES
) -> float:
    """Estimate the memory of ``matrices`` dense float64 pairwise arrays.

    Parameters
    ----------
    n_rows : int
        Number of observations.
    matrices : int
        Number of simultaneous ``n_rows x n_rows`` arrays.

    Returns
    -------
    float
        Estimate in MiB.
    """
    return matrices * BYTES_PER_FLOAT64 * n_rows * n_rows / _MIB


def guard_pairwise(
    n_rows: int,
    max_mb: float,
    matrices: int = DEFAULT_PAIRWISE_MATRICES,
) -> None:
    """Refuse work whose pairwise estimate exceeds ``max_mb``.

    This is a preventive estimate, not a cap on the process RSS.

    Raises
    ------
    MemoryBudgetError
        If the estimate exceeds the budget.
    """
    estimated = estimate_pairwise_mib(n_rows, matrices)
    if estimated > max_mb:
        raise MemoryBudgetError(
            f"Pairwise workspace estimate {estimated:.1f} MiB exceeds "
            f"{max_mb} MiB."
        )


class Metric(Protocol):
    """Strategy interface of a fitted reference metric."""

    def transform(self, frame: pd.DataFrame) -> GeometryValues:
        """Map raw rows into the metric's representation."""
        ...

    def distances(
        self, left: GeometryValues, right: GeometryValues | None = None
    ) -> np.ndarray:
        """Return the dense distance matrix between two representations."""
        ...


_SCALERS: dict[str, Callable[[], Any]] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
}


class EuclideanMetric:
    """Euclidean distance after an optional fitted scaler.

    Parameters
    ----------
    columns : sequence of hashable
        Numeric columns, in order.
    scaling : {"standard", "robust", "none"}
        Scaler fitted on the training rows.
    """

    def __init__(self, columns: Sequence[Any], scaling: str) -> None:
        self.columns = list(columns)
        self.scaling = scaling
        self.scaler_: Any = None

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit the scaler, if any, on ``frame``."""
        factory = _SCALERS.get(self.scaling)
        if factory is not None:
            self.scaler_ = factory().fit(self._values(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return scaled numeric values as a new array."""
        values = self._values(frame)
        if self.scaler_ is None:
            return values.copy()
        return np.asarray(self.scaler_.transform(values))

    def distances(
        self, left: GeometryValues, right: GeometryValues | None = None
    ) -> np.ndarray:
        """Return euclidean distances between rows of ``left``/``right``."""
        return np.asarray(pairwise_distances(left, right, metric="euclidean"))

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.columns].to_numpy(dtype=float)


class GowerMetric:
    """Weighted Gower distance over numeric, nominal, ordinal and binary data.

    For rows ``i`` and ``j`` the distance is

    ``sum_c w_c * delta_ijc * d_ijc / sum_c w_c * delta_ijc``

    where ``w_c`` is the variable weight (1 by default) and ``d_ijc`` the
    per-variable dissimilarity in [0, 1]:

    - numeric: ``|x_i - x_j| / range``, with the training range; values
      outside it are clipped so each contribution stays in [0, 1];
    - nominal: 0 on match, 1 on mismatch;
    - ordinal: absolute difference of level positions divided by
      ``levels - 1`` of the declared scale;
    - asymmetric binary: 0 when both are 1, 1 when they differ.

    ``delta_ijc`` is 0 for an asymmetric binary joint absence (both 0) and
    when either value is missing (pairwise deletion, only reachable with
    ``GeometrySpec(missing="pairwise")``), and 1 otherwise. Two rows with
    no comparable variable are rejected by default. Unknown nominal levels,
    undeclared ordinal levels and non-binary values raise an error at
    transform time.

    Parameters
    ----------
    numeric, categorical, binary : sequence of hashable
        Nonconstant numeric, nominal and asymmetric binary columns.
    ordinal : mapping of hashable to sequence
        Nonconstant ordinal columns and their declared ordered levels.
    weights : mapping of hashable to float or None
        Positive weight per column; missing columns weigh 1.
    """

    def __init__(
        self,
        numeric: Sequence[Any],
        categorical: Sequence[Any],
        ordinal: Mapping[Any, Sequence[Any]] | None = None,
        binary: Sequence[Any] = (),
        weights: Mapping[Any, float] | None = None,
        vectorized: bool = False,
        incomparable: str = "error",
        min_comparable_share: float = 0.0,
    ) -> None:
        if incomparable not in ("error", "max", "zero"):
            raise ValueError("incomparable must be error, max or zero")
        if not 0 <= min_comparable_share <= 1:
            raise ValueError("min_comparable_share must lie in [0, 1]")
        self.incomparable = incomparable
        self.min_comparable_share = float(min_comparable_share)
        self.vectorized = bool(vectorized)
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.ordinal_levels = {
            column: tuple(levels)
            for column, levels in (ordinal or {}).items()
        }
        self.binary = list(binary)
        self.weights = dict(weights or {})
        self.ranges_: dict[Any, float] = {}
        self.levels_: dict[Any, set[Any]] = {}
        self.bounds_: dict[Any, tuple[float, float]] = {}
        self.category_codes_: dict[Any, dict[Any, int]] = {}

    @property
    def ordinal(self) -> list[Any]:
        """Ordinal columns, in declaration order."""
        return list(self.ordinal_levels)

    @property
    def columns(self) -> list[Any]:
        """All columns contributing to the distance."""
        return self.numeric + self.categorical + self.ordinal + self.binary

    def weight(self, column: Any) -> float:
        """Weight of ``column`` (1 when not declared)."""
        return float(self.weights.get(column, 1.0))

    def fit(self, frame: pd.DataFrame) -> Self:
        """Record numeric ranges and the known nominal levels."""
        self.ranges_ = {
            c: float(frame[c].max() - frame[c].min()) for c in self.numeric
        }
        self.levels_ = {
            c: set(frame[c].dropna()) for c in self.categorical
        }
        self.bounds_ = {
            c: (float(frame[c].min()), float(frame[c].max()))
            for c in self.numeric
        }
        self.category_codes_ = {
            c: {level: i for i, level in
                enumerate(pd.unique(frame[c].dropna()))}
            for c in self.categorical
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the contributing columns in metric coding.

        Numeric and nominal values are copied unchanged. Ordinal values
        become their position divided by ``levels - 1`` (in [0, 1]) and
        binary values become integers 0/1.

        Raises
        ------
        ValueError
            If a nominal level was unseen in training, an ordinal value is
            not a declared level, or a binary value is not 0/1.
        """
        for column, levels in self.levels_.items():
            if not set(frame[column].dropna()).issubset(levels):
                raise ValueError(
                    f"Unknown category in {column!r}; the policy for unseen "
                    "levels is 'error'."
                )
        coded = frame[self.columns].copy()
        for column, declared in self.ordinal_levels.items():
            positions = {level: index for index, level in enumerate(declared)}
            scores = coded[column].map(positions)
            if (scores.isna() & coded[column].notna()).any():
                raise ValueError(
                    f"Ordinal column {column!r} has undeclared levels."
                )
            coded[column] = scores.to_numpy(dtype=float) / (len(declared) - 1)
        for column in self.binary:
            series = coded[column]
            observed = series.dropna().to_numpy()
            if not np.isin(observed, (0, 1)).all():
                raise ValueError(
                    f"Binary column {column!r} must contain only 0/1."
                )
            if series.isna().any():
                coded[column] = series.astype(float)
            else:
                coded[column] = series.to_numpy().astype(int)
        return coded

    def distances(
        self, left: GeometryValues, right: GeometryValues | None = None
    ) -> np.ndarray:
        """Weighted Gower under stable or opt-in compiled backend.

        The default implementation preserves historical summation order.
        With ``vectorized=True`` complete cases use SciPy compiled distance
        kernels; missing or out-of-range values use the stable path.
        Missing common features or insufficient coverage follows the explicit
        ``incomparable`` policy, never an implicit perfect match.
        """
        if not isinstance(left, pd.DataFrame):
            raise TypeError("Gower distances require DataFrame inputs.")
        if getattr(self, "vectorized", False):
            return self._vectorized_distances(left, right)
        return self._stable_distances(left, right)

    def _stable_distances(
        self, left: pd.DataFrame, right: GeometryValues | None = None
    ) -> np.ndarray:
        """Reference-compatible arithmetic with feature-order reduction."""
        other = left if right is None else right
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Gower distances require DataFrame inputs.")
        n, m = len(left), len(other)
        shape = (n, m)
        numerator = np.zeros(shape, dtype=float)
        denominator = np.zeros(shape, dtype=float)
        # A broadcasted block holds at most ~4 million elements. A small
        # block also bounds the number of object comparisons for nominals.
        block_size = max(1, min(16, 4_000_000 // max(1, n * m)))
        complete = (
            not left[self.columns].isna().to_numpy().any()
            and not other[self.columns].isna().to_numpy().any()
        )
        # For complete numeric, categorical and ordinal variables the
        # denominator is constant; this retains the exact Gower semantics.
        if complete:
            denominator += sum(
                self.weight(c)
                for c in self.numeric + self.categorical + self.ordinal
            )

        for columns, kind in (
            (self.numeric, "numeric"),
            (self.categorical, "categorical"),
            (self.ordinal, "ordinal"),
            (self.binary, "binary"),
        ):
            for start in range(0, len(columns), block_size):
                cols = columns[start:start + block_size]
                weights = np.asarray([self.weight(c) for c in cols])
                if kind == "categorical":
                    a = left[cols].to_numpy(dtype=object)[:, None, :]
                    b = other[cols].to_numpy(dtype=object)[None, :, :]
                    observed = ~(pd.isna(a) | pd.isna(b))
                    if not complete:
                        a = np.where(pd.isna(a), None, a)
                        b = np.where(pd.isna(b), None, b)
                    mismatch = (a != b) & observed
                    # Preserve feature-order accumulation. The medoid
                    # optimizer may branch on a last-bit numerical tie;
                    # reductions across features can alter that branch.
                    for k, weight in enumerate(weights):
                        numerator += weight * mismatch[:, :, k]
                        if not complete:
                            denominator += weight * observed[:, :, k]
                    continue

                a = left[cols].to_numpy(dtype=float)[:, None, :]
                b = other[cols].to_numpy(dtype=float)[None, :, :]
                observed = ~(np.isnan(a) | np.isnan(b))
                if kind == "binary":
                    aa = np.nan_to_num(a)
                    bb = np.nan_to_num(b)
                    mismatch = (aa != bb) & observed
                    active = ((aa + bb > 0) & observed)
                    for k, weight in enumerate(weights):
                        numerator += weight * mismatch[:, :, k]
                        denominator += weight * active[:, :, k]
                    continue
                if kind == "numeric":
                    # Work per-column on 2-D views so the same NumPy ufunc
                    # kernels and last-bit arithmetic as the reference run.
                    for k, weight in enumerate(weights):
                        a2, b2 = a[:, :, k], b[:, :, k]
                        value = np.minimum(
                            np.abs(a2 - b2) / self.ranges_[cols[k]], 1.0
                        )
                        if complete:
                            numerator += weight * value
                        else:
                            numerator += weight * np.where(
                                observed[:, :, k], value, 0.0
                            )
                            denominator += weight * observed[:, :, k]
                    continue
                differences = np.abs(a - b)
                for k, weight in enumerate(weights):
                    if complete:
                        numerator += weight * differences[:, :, k]
                    else:
                        numerator += weight * np.where(
                            observed[:, :, k], differences[:, :, k], 0.0
                        )
                        denominator += weight * observed[:, :, k]
        return self._finalize(numerator, denominator, same_rows=other is left)

    def _vectorized_distances(
        self, left: pd.DataFrame, right: GeometryValues | None
    ) -> np.ndarray:
        """C-level weighted Gower using SciPy distance kernels and BLAS.

        Only dense complete cases take this path. Missing values and
        out-of-training-range values fall back to exact stable arithmetic,
        preserving pairwise deletion and per-variable clipping semantics.
        Floating-point summation order differs, so exact-tie model paths
        may differ; this backend is intentionally opt-in.
        """
        other = left if right is None else right
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Gower distances require DataFrame inputs.")
        if (left[self.columns].isna().to_numpy().any()
                or other[self.columns].isna().to_numpy().any()):
            return self._stable_distances(left, other)
        for col, (minimum, maximum) in self.bounds_.items():
            if (left[col].lt(minimum).any() or left[col].gt(maximum).any()
                    or other[col].lt(minimum).any()
                    or other[col].gt(maximum).any()):
                return self._stable_distances(left, other)
        shape = (len(left), len(other))
        numerator = np.zeros(shape, dtype=float)
        denominator = np.zeros(shape, dtype=float)
        if self.numeric:
            cols = self.numeric
            ranges = np.array([self.ranges_[c] for c in cols], dtype=float)
            keep = ranges > 0
            if np.any(keep):
                scale = np.array([self.weight(c) for c in cols])[keep]
                a = left[cols].to_numpy(dtype=float)[:, keep] / ranges[keep]
                b = other[cols].to_numpy(dtype=float)[:, keep] / ranges[keep]
                numerator += cdist(a, b, metric="cityblock", w=scale)
            denominator += sum(self.weight(c) for c in cols)
        if self.categorical:
            cols = self.categorical
            a = np.column_stack([
                left[c].map(self.category_codes_[c]).to_numpy(dtype=float)
                for c in cols
            ])
            b = np.column_stack([
                other[c].map(self.category_codes_[c]).to_numpy(dtype=float)
                for c in cols
            ])
            weights = np.array([self.weight(c) for c in cols])
            # SciPy Hamming is normalised by the sum of supplied weights.
            numerator += (
                cdist(a, b, metric="hamming", w=weights) * weights.sum()
            )
            denominator += weights.sum()
        if self.ordinal:
            cols = self.ordinal
            a = left[cols].to_numpy(dtype=float)
            b = other[cols].to_numpy(dtype=float)
            weights = np.array([self.weight(c) for c in cols])
            numerator += cdist(a, b, metric="cityblock", w=weights)
            denominator += weights.sum()
        if self.binary:
            cols = self.binary
            weights = np.array([self.weight(c) for c in cols])
            a = left[cols].to_numpy(dtype=float)
            b = other[cols].to_numpy(dtype=float)
            left_sum = (a * weights).sum(axis=1)[:, None]
            right_sum = (b * weights).sum(axis=1)[None, :]
            shared = (a * weights) @ b.T
            numerator += left_sum + right_sum - 2 * shared
            denominator += left_sum + right_sum - shared
        # Roundoff in the dot product can produce -1e-15 for exact zeros.
        np.maximum(numerator, 0.0, out=numerator)
        np.maximum(denominator, 0.0, out=denominator)
        distance = self._finalize(
            numerator, denominator, same_rows=other is left
        )
        return np.clip(distance, 0.0, 1.0, out=distance)

    def _finalize(
        self, numerator: np.ndarray, denominator: np.ndarray,
        *, same_rows: bool
    ) -> np.ndarray:
        """Apply explicit comparability policy before division.

        A self-comparison always has distance zero, even for a row with
        asymmetric-binary joint absences (the diagonal is not evidence).
        """
        total_weight = sum(self.weight(c) for c in self.columns)
        if total_weight <= 0:
            raise ValueError("Gower needs at least one positive weight.")
        invalid = (denominator <= 0) | (
            denominator / total_weight < self.min_comparable_share
        )
        if same_rows:
            np.fill_diagonal(invalid, False)
        if invalid.any() and self.incomparable == "error":
            i, j = np.argwhere(invalid)[0]
            raise ValueError(
                "No sufficient comparable features for Gower pair "
                f"({int(i)}, {int(j)}); choose incomparable='max' "
                "to penalize or 'zero' for legacy behavior."
            )
        result = np.zeros_like(numerator, dtype=float)
        np.divide(
            numerator, denominator, out=result,
            where=(~invalid) & (denominator > 0),
        )
        if self.incomparable == "max":
            result[invalid] = 1.0
        if same_rows:
            np.fill_diagonal(result, 0.0)
        return result

    def comparable_share(
        self, left: pd.DataFrame, right: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Fraction of total Gower feature weight comparable per pair.

        This is a diagnostic, NOT a distance. In particular, a 0 share
        must never be interpreted as a 0 distance.
        """
        other = left if right is None else right
        weights = np.zeros((len(left), len(other)), dtype=float)
        for col in self.columns:
            a = left[col].notna().to_numpy()
            b = other[col].notna().to_numpy()
            comparable = a[:, None] & b[None, :]
            if col in self.binary:
                x = left[col].fillna(0).to_numpy(dtype=float)
                y = other[col].fillna(0).to_numpy(dtype=float)
                comparable &= (x[:, None] != 0) | (y[None, :] != 0)
            weights += self.weight(col) * comparable
        return weights / sum(self.weight(col) for col in self.columns)

    def distance_blocks(
        self, left: pd.DataFrame, right: pd.DataFrame | None = None,
        *, block_rows: int = 256
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, stop, distances)`` without an n-by-m allocation.

        Consumers process blocks incrementally; assembling every block
        into a full matrix defeats the memory-saving purpose.
        """
        if isinstance(block_rows, bool) or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        other = left if right is None else right
        for start in range(0, len(left), block_rows):
            stop = min(start + block_rows, len(left))
            part = left.iloc[start:stop]
            if other is left:
                # Compare to left, but self pairs must be masked before
                # incomparable detection. Assemble the block by columns
                # around its diagonal to make self-block explicit.
                parts = []
                if start:
                    parts.append(self.distances(part, other.iloc[:start]))
                parts.append(self.distances(part))
                if stop < len(other):
                    parts.append(self.distances(part, other.iloc[stop:]))
                block = np.concatenate(parts, axis=1)
            else:
                block = self.distances(part, other)
            yield start, stop, block

    def _pairwise_distances(
        self, left: pd.DataFrame, other: pd.DataFrame
    ) -> np.ndarray:
        """Compatibility shim; both paths share the same vectorised code."""
        return self.distances(left, other)


class ReferenceGeometry:
    """Fitted reference representation and distance.

    Constant features are excluded at fit time. Euclidean geometry requires
    numeric features only; Gower handles nominal, ordinal, asymmetric binary
    and mixed tables.

    Parameters
    ----------
    spec : GeometrySpec
        Geometry kind and scaling.
    schema : Schema
        Resolved column roles.
    """

    def __init__(self, spec: GeometrySpec, schema: Schema) -> None:
        self.spec = spec
        self.schema = schema

    def fit(self, frame: pd.DataFrame) -> Self:
        """Select nonconstant features and fit the metric on ``frame``.

        Raises
        ------
        ValueError
            If every declared feature is constant, or a weight names a
            column that is not a declared feature.
        CapabilityError
            If euclidean geometry is requested with non-numeric features.
        """
        schema = self.schema
        weights = dict(self.spec.weights)
        unknown = [c for c in weights if c not in schema.features]
        if unknown:
            raise ValueError(
                f"Weights name columns that are not features: {unknown}."
            )
        self.columns_ = [c for c in schema.features if frame[c].nunique() > 1]
        if not self.columns_:
            raise ValueError("All declared features are constant.")
        kept = set(self.columns_)
        self.numeric_ = [c for c in schema.numeric if c in kept]
        self.categorical_ = [c for c in schema.categorical if c in kept]
        self.ordinal_ = [c for c in schema.ordinal_columns if c in kept]
        self.binary_ = [c for c in schema.binary if c in kept]
        self.metric_: EuclideanMetric | GowerMetric
        if self.spec.kind == "euclidean":
            if schema.nonnumeric:
                raise CapabilityError(
                    "Euclidean geometry requires numeric features. Use Gower "
                    "for nominal, ordinal, binary or mixed data."
                )
            self.metric_ = EuclideanMetric(self.columns_, self.spec.scaling)
        else:
            levels = schema.ordinal_levels
            self.metric_ = GowerMetric(
                self.numeric_,
                self.categorical_,
                {c: levels[c] for c in self.ordinal_},
                self.binary_,
                weights,
                vectorized=self.spec.gower_backend == "vectorized",
                incomparable=self.spec.incomparable,
                min_comparable_share=self.spec.min_comparable_share,
            )
        self.metric_.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> GeometryValues:
        """Map rows into the fitted reference representation."""
        return self.metric_.transform(frame)

    def distances(
        self, left: GeometryValues, right: GeometryValues | None = None
    ) -> np.ndarray:
        """Return distances between transformed representations."""
        return self.metric_.distances(left, right)

    def distance_blocks(
        self, left: GeometryValues, right: GeometryValues | None = None,
        *, block_rows: int = 256
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield row blocks without materializing a full matrix."""
        if isinstance(self.metric_, GowerMetric):
            yield from self.metric_.distance_blocks(
                left, right, block_rows=block_rows
            )
        else:
            other = left if right is None else right
            for start in range(0, len(left), block_rows):
                stop = min(start + block_rows, len(left))
                yield start, stop, self.distances(left[start:stop], other)
