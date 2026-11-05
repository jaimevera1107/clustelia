"""Experiment results and fitted solutions.

Results expose copies of their internal arrays so that callers cannot
alter the recorded evidence by mutating returned objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .adapters import FittedCandidate, Membership
from .config import Schema
from .records import CandidateRow
from .schema import validate_table

__all__ = ["PROFILE_QUANTILES", "ExperimentResult", "Recommendation",
           "Solution"]

PROFILE_QUANTILES: tuple[float, float] = (0.25, 0.75)
"""Lower and upper quantiles reported by :meth:`Solution.describe`."""


@dataclass(frozen=True)
class Recommendation:
    """Identifier of the candidate preferred by the selection policy."""

    candidate_id: str


class Solution:
    """One fitted candidate and its training labels.

    Solutions of the same experiment share a single read-only reference to
    the validated training table instead of holding one copy each.

    Parameters
    ----------
    candidate_id : str
        Stable candidate identifier.
    fitted : FittedCandidate
        Fitted pipeline (never refitted by the solution).
    frame : pandas.DataFrame
        Validated training table, owned by the experiment.
    schema : Schema
        Resolved column roles.
    """

    def __init__(
        self,
        candidate_id: str,
        fitted: FittedCandidate,
        frame: pd.DataFrame,
        schema: Schema,
    ) -> None:
        self.candidate_id = candidate_id
        self.pipeline_ = fitted
        self.schema_ = schema
        self._frame = frame
        self._labels = np.asarray(fitted.labels_).copy()

    @property
    def labels_(self) -> np.ndarray:
        """Copy of the training labels (``-1`` denotes noise)."""
        return self._labels.copy()

    @property
    def membership_(self) -> Membership | None:
        """Copy of the soft assignment, if the adapter provides one."""
        return deepcopy(self.pipeline_.membership_)

    @property
    def structure_(self) -> dict[str, Any] | None:
        """Copy of the Ward tree or OPTICS reachability, if available."""
        return deepcopy(self.pipeline_.structure_)

    def assign(self, X: Any) -> np.ndarray:
        """Assign new rows with the fitted pipeline.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            New rows with exactly the training columns, in order.

        Raises
        ------
        CapabilityError
            If the adapter is transductive.
        ValueError
            If the rows violate the training schema.
        """
        frame = validate_table(
            X, self.schema_, self._frame.columns, min_rows=1,
            allow_missing=(
                self.pipeline_.geometry_spec.missing == "pairwise"
            ),
        ).frame
        return self.pipeline_.assign(frame)

    def sizes_table(self) -> pd.DataFrame:
        """Cluster sizes, coverage and noise in a readable table."""
        counts = pd.Series(self._labels).value_counts().sort_index()
        return pd.DataFrame({
            "cluster": counts.index.astype(int),
            "size": counts.to_numpy(dtype=int),
            "share": counts.to_numpy(dtype=float) / len(self._labels),
            "is_noise": counts.index.to_numpy() == -1,
        }).reset_index(drop=True)

    def profile_table(self) -> pd.DataFrame:
        """Long, tidy per-cluster feature profile in original data units.

        Numeric cells use median/IQR; nominal/ordinal the most common
        observed level and share; asymmetric binary the share equal to 1.
        Missingness and group sizes are explicit. No variables are
        standardised or silently reordered for this description.
        """
        rows: list[dict[str, Any]] = []
        for label in np.unique(self._labels):
            frame = self._frame.iloc[np.flatnonzero(self._labels == label)]
            for column in self.schema_.features:
                values = frame[column]
                observed = values.dropna()
                kind = (
                    "numeric" if column in self.schema_.numeric else
                    "ordinal" if column in self.schema_.ordinal_columns else
                    "binary" if column in self.schema_.binary else
                    "categorical"
                )
                row: dict[str, Any] = {
                    "cluster": int(label), "is_noise": bool(label < 0),
                    "size": len(frame), "feature": str(column),
                    "type": kind,
                    "missing_share": float(values.isna().mean()),
                    "median": None, "q25": None, "q75": None,
                    "top_value": None, "top_share": None,
                    "share_one": None,
                    "global_median": None,
                    "median_delta": None,
                    "global_top_share": None,
                    "top_lift": None,
                    "global_share_one": None,
                    "share_one_delta": None,
                }
                baseline = self._frame[column].dropna()
                if kind == "numeric" and len(observed):
                    global_median = float(baseline.median())
                    median = float(observed.median())
                    row.update(median=median,
                               q25=float(observed.quantile(.25)),
                               q75=float(observed.quantile(.75)),
                               global_median=global_median,
                               median_delta=median - global_median)
                elif kind == "binary" and len(observed):
                    share = float(observed.astype(int).mean())
                    global_share = float(baseline.astype(int).mean())
                    row.update(share_one=share,
                               global_share_one=global_share,
                               share_one_delta=share - global_share)
                elif len(observed):
                    counts = observed.value_counts(normalize=True)
                    row["top_value"] = counts.index[0]
                    row["top_share"] = float(counts.iloc[0])
                    global_share = float(
                        (baseline == counts.index[0]).mean()
                    )
                    row["global_top_share"] = global_share
                    row["top_lift"] = (
                        float(counts.iloc[0]) / global_share
                        if global_share > 0 else None
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def plot_sizes(self, *, ax: Any = None) -> Any:
        """Plot cluster sizes (noise is labelled separately).

        Requires the optional ``viz`` extra (matplotlib). Returns Axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot_sizes requires matplotlib: pip install clustelia[viz]"
            ) from exc
        if ax is None:
            _, ax = plt.subplots()
        table = self.sizes_table()
        labels = ["Noise" if c == -1 else f"Cluster {c}"
                  for c in table["cluster"]]
        ax.bar(labels, table["size"])
        ax.set(xlabel="Partition", ylabel="Observations",
               title="Cluster sizes")
        ax.tick_params(axis="x", labelrotation=35)
        return ax

    def plot_profile(
        self, feature: Any, *, ax: Any = None, max_categories: int = 12
    ) -> Any:
        """Visual comparison of a feature between clusters.

        Numeric: per-cluster boxplots in input units. Nominal/ordinal:
        conditional category shares (top ``max_categories`` across all
        rows). Asymmetric binary: within-cluster observed share of ones.
        Noise is always identified separately. Returns matplotlib Axes.
        """
        if feature not in self.schema_.features:
            raise ValueError(
                f"Feature {feature!r} is not a clustering feature."
            )
        if max_categories < 1:
            raise ValueError("max_categories must be positive.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install clustelia[viz] to plot.") from exc
        if ax is None:
            _, ax = plt.subplots()
        data = self._frame[[feature]].copy()
        data["_cluster"] = self._labels
        groups = sorted(data["_cluster"].unique())
        labels = ["Noise" if g == -1 else f"Cluster {g}" for g in groups]
        if feature in self.schema_.numeric:
            samples = [data.loc[data["_cluster"] == g, feature]
                       .dropna().to_numpy(dtype=float) for g in groups]
            if any(len(a) for a in samples):
                ax.boxplot(samples, positions=range(1, len(labels) + 1))
                ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.set(ylabel=str(feature), title=f"Distribution: {feature}")
        elif feature in self.schema_.binary:
            shares = [data.loc[data["_cluster"] == g, feature]
                      .dropna().astype(float).mean() for g in groups]
            ax.bar(labels, shares)
            ax.set(ylabel="Share of ones", ylim=(0, 1),
                   title=f"Binary feature: {feature}")
        else:
            top = data[feature].value_counts().index[:max_categories]
            counts = pd.crosstab(data["_cluster"], data[feature],
                                 normalize="index")
            fractions = counts.reindex(index=groups, columns=top,
                                       fill_value=0)
            bottom = np.zeros(len(groups))
            for level in top:
                heights = fractions[level].to_numpy(dtype=float)
                ax.bar(labels, heights, bottom=bottom, label=str(level))
                bottom += heights
            ax.legend(title=str(feature), bbox_to_anchor=(1.05, 1),
                      loc="upper left")
            ax.set(ylabel="Observed share", ylim=(0, 1),
                   title=f"Distribution: {feature}")
        ax.tick_params(axis="x", labelrotation=35)
        return ax

    def distribution_table(
        self, *, include_missing: bool = True
    ) -> pd.DataFrame:
        """Long-form categorical frequencies, proportions and enrichment.

        Descriptive only: the labels were constructed from these features,
        so these are NOT out-of-sample tests of association.
        """
        rows: list[dict[str, Any]] = []
        categorical = list(self.schema_.categorical) + list(
            self.schema_.ordinal_columns
        ) + list(self.schema_.binary)
        for col in categorical:
            baseline = self._frame[col]
            global_counts = baseline.value_counts(dropna=not include_missing)
            for group in np.unique(self._labels):
                values = baseline.iloc[np.flatnonzero(self._labels == group)]
                counts = values.value_counts(dropna=not include_missing)
                for level, n in counts.items():
                    # Compare NaN without equality because NaN != NaN.
                    total = int(global_counts.loc[level])
                    share = int(n) / len(values)
                    global_share = total / len(baseline)
                    rows.append({
                        "cluster": int(group), "is_noise": bool(group == -1),
                        "feature": str(col),
                        "level": "<missing>" if pd.isna(level) else str(level),
                        "count": int(n), "cluster_size": len(values),
                        "share": share, "global_share": global_share,
                        "lift": share / global_share if global_share else None,
                    })
        return pd.DataFrame(rows, columns=[
            "cluster", "is_noise", "feature", "level", "count",
            "cluster_size", "share", "global_share", "lift",
        ])

    def feature_importance_table(self) -> pd.DataFrame:
        """Post-hoc descriptive differentiation of the fitted clusters.

        Numeric features: between-group explained variance (eta squared).
        Other features: bias-corrected squared Cramér V. This reduces
        spurious differentiation from high-cardinality categories.
        No significance tests or generalisation claims are made because
        the clustering algorithm uses these very same variables.
        Noise is excluded from comparison.
        """
        from scipy.stats import chi2_contingency
        keep = self._labels >= 0
        labels = self._labels[keep]
        records = []
        for col in self.schema_.features:
            values = self._frame[col].to_numpy()[keep]
            valid = pd.notna(values)
            groups = labels[valid]
            observed = values[valid]
            score = 0.0
            if len(observed) and len(np.unique(groups)) > 1:
                if col in self.schema_.numeric:
                    x = np.asarray(observed, dtype=float)
                    overall = float(x.mean())
                    ss_total = float(np.square(x - overall).sum())
                    ss_between = sum(
                        int(np.sum(groups == g)) *
                        (float(x[groups == g].mean()) - overall) ** 2
                        for g in np.unique(groups)
                    )
                    score = ss_between / ss_total if ss_total else 0.0
                    method = "eta_squared"
                else:
                    contingency = pd.crosstab(
                        pd.Series(groups), pd.Series(observed)
                    )
                    r, c = contingency.shape
                    if min(r, c) > 1:
                        chi2 = chi2_contingency(
                            contingency, correction=False
                        )[0]
                        n = len(observed)
                        phi2 = chi2 / n
                        phi2_corr = max(
                            0.0, phi2 - (r - 1) * (c - 1) / (n - 1)
                        )
                        r_corr = r - (r - 1) ** 2 / (n - 1)
                        c_corr = c - (c - 1) ** 2 / (n - 1)
                        denom = min(r_corr - 1, c_corr - 1)
                        score = phi2_corr / denom if denom > 0 else 0.0
                    method = "cramers_v_squared_bias_corrected"
            else:
                method = ("eta_squared" if col in self.schema_.numeric
                          else "cramers_v_squared_bias_corrected")
            records.append({
                "feature": str(col), "measure": method,
                "differentiation": float(np.clip(score, 0, 1)),
                "observed_rows": int(valid.sum()),
                "missing_share": 1 - float(valid.mean()),
            })
        return pd.DataFrame(records).sort_values(
            "differentiation", ascending=False, kind="stable"
        ).reset_index(drop=True)

    def diagnostics_table(self) -> pd.DataFrame:
        """Partition balance and missingness; not validity tests."""
        table = self.sizes_table()
        assigned = table[~table["is_noise"]]
        n = int(assigned["size"].sum())
        shares = assigned["size"].to_numpy(dtype=float) / max(n, 1)
        entropy = -float(np.sum(shares * np.log(shares))) if n else None
        return pd.DataFrame([{
            "n_rows": len(self._labels), "n_clusters": len(assigned),
            "assigned_share": n / len(self._labels),
            "noise_share": 1 - n / len(self._labels),
            "smallest_cluster": (int(assigned["size"].min())
                                 if len(assigned) else None),
            "largest_cluster": (int(assigned["size"].max())
                                if len(assigned) else None),
            "balance_ratio": (float(assigned["size"].min() /
                                    assigned["size"].max())
                              if len(assigned) else None),
            "entropy": entropy,
        }])

    def plot_importance(self, *, ax: Any = None, top: int = 15) -> Any:
        """Bar plot of descriptive differentiation (not causal importance)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install clustelia[viz] to plot.") from exc
        table = self.feature_importance_table().head(top)
        if ax is None:
            _, ax = plt.subplots()
        ax.barh(table["feature"][::-1], table["differentiation"][::-1])
        ax.set(xlabel="Post-hoc differentiation (0–1)",
               title="Cluster differentiation — descriptive only", xlim=(0, 1))
        return ax

    def plot_heatmap(self, *, ax: Any = None) -> Any:
        """Heatmap of numeric cluster means in overall SD units."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install clustelia[viz] to plot.") from exc
        cols = list(self.schema_.numeric)
        if not cols:
            raise ValueError("Heatmap requires numeric features.")
        scaled = self._frame[cols].astype(float)
        sd = scaled.std(ddof=0).replace(0, np.nan)
        standardized = (scaled - scaled.mean()) / sd
        frame = standardized.copy()
        frame["_cluster"] = self._labels
        heat = frame.groupby("_cluster", sort=True).mean().T.fillna(0)
        if ax is None:
            _, ax = plt.subplots(figsize=(max(5, len(heat.columns) * 1.5),
                                           max(3, len(cols) * .35)))
        image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm",
                          vmin=-2, vmax=2)
        ax.set_xticks(range(len(heat.columns)), heat.columns.map(str))
        ax.set_yticks(range(len(heat.index)), heat.index.map(str))
        ax.set(xlabel="Cluster (-1 = noise)",
               title="Mean differences, scaled by overall SD")
        ax.figure.colorbar(image, ax=ax, label="Standard deviations")
        return ax

    def plot_projection(
        self, *, ax: Any = None, max_points: int = 2000, random_state: int = 0
    ) -> Any:
        """Descriptive 2D PCA projection of numeric/one-hot features.

        This visualisation cannot validate cluster separation. A fixed
        sample is used solely for readability; no model is refitted.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install clustelia[viz] to plot.") from exc
        if max_points < 2:
            raise ValueError("max_points must be >= 2")
        indices = np.arange(len(self._frame))
        if len(indices) > max_points:
            rng = np.random.default_rng(random_state)
            indices = np.sort(rng.choice(indices, max_points, replace=False))
        data = pd.get_dummies(
            self._frame.iloc[indices][list(self.schema_.features)],
            dummy_na=True, dtype=float
        ).fillna(0).astype(float)
        if data.shape[1] < 2:
            raise ValueError("Projection needs at least 2 encoded features")
        if data.shape[1] > 1000:
            raise ValueError("Projection exceeds 1000 encoded features")
        scaled = StandardScaler().fit_transform(data)
        projected = PCA(
            n_components=2, random_state=random_state
        ).fit_transform(scaled)
        if ax is None:
            _, ax = plt.subplots()
        for group in np.unique(self._labels[indices]):
            use = self._labels[indices] == group
            ax.scatter(projected[use, 0], projected[use, 1], s=12,
                       alpha=.65, label="Noise" if group == -1 else str(group))
        ax.set(xlabel="PCA component 1", ylabel="PCA component 2",
               title="Exploratory projection — not validation")
        ax.legend(title="Cluster")
        return ax

    def describe(self) -> list[dict[str, Any]]:
        """Profile every cluster in the original variables.

        Returns
        -------
        list of dict
            Size, noise flag, numeric quartiles/median, categorical
            proportions, ordinal lower-median level and proportions in
            declared order, and the share of ones of each binary variable,
            per label. With missing values (Gower pairwise deletion) every
            summary uses the observed values and a ``"missing"`` entry gives
            the missing fraction of each variable that has any.
        """
        low, high = PROFILE_QUANTILES
        profiles: list[dict[str, Any]] = []
        for label in np.unique(self._labels):
            group = self._frame.iloc[np.flatnonzero(self._labels == label)]
            numeric = {
                str(c): {
                    "q25": float(group[c].quantile(low)),
                    "median": float(group[c].median()),
                    "q75": float(group[c].quantile(high)),
                }
                for c in self.schema_.numeric
            }
            categorical = {
                str(c): group[c].value_counts(normalize=True).to_dict()
                for c in self.schema_.categorical
            }
            profiles.append({
                "label": int(label),
                "size": len(group),
                "is_noise": bool(label < 0),
                "numeric": numeric,
                "categorical": categorical,
                "ordinal": {
                    str(c): _ordinal_profile(group[c], levels)
                    for c, levels in self.schema_.ordinal
                },
                "binary": {
                    str(c): _binary_share(group[c])
                    for c in self.schema_.binary
                },
            })
            missing = group[list(self.schema_.features)].isna().mean()
            if missing.any():
                profiles[-1]["missing"] = {
                    str(c): float(v) for c, v in missing.items() if v > 0
                }
        return profiles


def _binary_share(values: pd.Series) -> float:
    """Share of ones among observed values (NaN when none observed)."""
    observed = values.dropna()
    return float(observed.astype(int).mean()) if len(observed) else np.nan


def _ordinal_profile(values: pd.Series, levels: tuple) -> dict[str, Any]:
    """Lower-median level and level proportions in declared order."""
    observed = values.dropna()
    positions = observed.map({level: i for i, level in enumerate(levels)})
    ordered = np.sort(positions.to_numpy(dtype=int))
    shares = observed.value_counts(normalize=True)
    return {
        "median": (levels[int(ordered[(len(ordered) - 1) // 2])]
                   if len(ordered) else None),
        "proportions": {
            level: float(shares.get(level, 0.0)) for level in levels
        },
    }


class ExperimentResult:
    """Outcome of one search: status, evidence and fitted solutions.

    Parameters
    ----------
    status : str
        ``"recommended"``, ``"ambiguous"``, ``"no_admissible_solution"``
        or ``"incomplete"``.
    rows : iterable of dict
        Per-candidate records.
    solutions : mapping
        Fitted solutions by candidate identifier.
    recommendation : str or None
        Preferred candidate identifier.
    metadata : mapping
        Policy, budget, environment, seeds, indices and costs.
    frontier : iterable of str
        Pareto-frontier candidate identifiers.
    """

    def __init__(
        self,
        status: str,
        rows: Iterable[CandidateRow],
        solutions: Mapping[str, Solution],
        recommendation: str | None,
        metadata: Mapping[str, Any],
        frontier: Iterable[str] = (),
    ) -> None:
        self.selection_status = status
        self.recommendation = (
            Recommendation(recommendation) if recommendation else None
        )
        self._rows: list[CandidateRow] = deepcopy(list(rows))
        self._solutions: dict[str, Solution] = dict(solutions)
        self.metadata: dict[str, Any] = deepcopy(dict(metadata))
        self.pareto_frontier = tuple(frontier)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Identifiers of every successfully fitted candidate."""
        return tuple(self._solutions)

    @property
    def fitted_algorithms(self) -> frozenset[str]:
        """Algorithms of every successfully fitted candidate."""
        return frozenset(
            s.pipeline_.spec.algorithm for s in self._solutions.values()
        )

    def _training_rows(self) -> int | None:
        """Number of training rows, or ``None`` without solutions."""
        for solution in self._solutions.values():
            return len(solution.labels_)
        return None

    def recommended_solution(self) -> Solution | None:
        """Return validated recommendation or None without fallback."""
        if self.recommendation is None:
            return None
        return self.select(self.recommendation.candidate_id)

    def save(self, path: str | Any) -> None:
        """Save this experiment with the trusted-versioned pickle envelope.

        Loading requires ``clustelia.load(path, trusted=True)``. Only load
        trusted files: pickle is not a secure interchange format.
        """
        from .persistence import save
        save(self, path)

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-friendly overview without claiming false certainty."""
        table = self.candidates_table()
        winner = (
            self.recommendation.candidate_id if self.recommendation else None
        )
        selected = (table.loc[table["candidate_id"] == winner]
                    if "candidate_id" in table else pd.DataFrame())
        record = selected.iloc[0] if len(selected) else None
        return {
            "status": self.selection_status,
            "objective": self.metadata.get("objective"),
            "validation_scope": self.metadata.get("validation_scope"),
            "effort": self.metadata.get("effort", "expert"),
            "policy": self.metadata.get("policy", {}).get("version"),
            "null_replicates_requested": self.metadata.get("policy", {}).get(
                "null_replicates"),
            "null_test_resolution": (
                1 / (1 + self.metadata["policy"]["null_replicates"])
                if self.metadata.get("policy", {}).get("null_replicates", 0)
                else None
            ),
            "null_replicates_completed": self.metadata.get(
                "structure_test", {}).get("replicates") if isinstance(
                    self.metadata.get("structure_test"), dict) else None,
            "rows": self._training_rows(),
            "candidates_evaluated": int(len(table)),
            "candidates_fitted": len(self._solutions),
            "fits_used": self.metadata.get("fits_used"),
            "recommended_candidate": winner,
            "algorithm": record["algorithm"] if record is not None else None,
            "n_clusters": (int(record["n_clusters"])
                           if record is not None
                           and pd.notna(record["n_clusters"])
                           else None),
            "coverage": (float(record["coverage"])
                         if record is not None and pd.notna(record["coverage"])
                         else None),
            "null_family": self.metadata.get("null_family"),
            "selected_k": self.metadata.get("selected_k"),
            "selected_k_at_boundary": self.metadata.get(
                "selected_k_at_boundary"
            ),
            "selection_warnings": list(
                self.metadata.get("selection_warnings", [])
            ),
            "incomplete_reasons": list(
                self.metadata.get("incomplete_reasons", [])
            ),
            "excluded_identifiers": list(
                self.metadata.get("diagnostics", {}).get(
                    "excluded_identifiers", []
                )
            ),
        }

    def ranking_table(self, top: int | None = 15) -> pd.DataFrame:
        """Read-only convenience comparison; higher scores are NOT universal.

        ``primary_score`` is only comparable within a common objective and
        geometry. No ranking replaces the selection/confirmation decision.
        """
        if top is not None and (isinstance(top, bool) or
                                not isinstance(top, int) or top < 1):
            raise ValueError("top must be a positive integer or None.")
        cols = ["candidate_id", "algorithm", "status", "confirmed",
                "n_clusters", "coverage", "silhouette", "primary_score",
                "rejection_reasons"]
        source = self.candidates_table()
        if source.empty:
            return pd.DataFrame(columns=["recommended", *cols])
        view = source.loc[:, [c for c in cols if c in source]].copy()
        chosen = (
            self.recommendation.candidate_id if self.recommendation else None
        )
        view.insert(0, "recommended", view["candidate_id"].eq(chosen))
        # Primary scores are oriented with larger being better, but
        # are comparable only within the same objective/representation.
        ordering = ["recommended", "confirmed", "primary_score",
                    "candidate_id"]
        descending = [False, False, False, True]
        view = view.sort_values(
            ordering, ascending=descending, na_position="last",
            kind="stable",
        )
        if top is not None:
            view = view.head(top)
        return view.reset_index(drop=True)

    def report(self, *, top: int = 5) -> str:
        """Markdown summary suitable for notebooks, reports and logs."""
        details = self.summary()
        algorithm = details["algorithm"] or "none"
        lines = [
            "# Clustelia experiment",
            f"- Selection: **{details['status']}**",
            f"- Objective: {details['objective']}",
            f"- Validation scope: {details['validation_scope']}",
            f"- Fit attempts: {details['fits_used']}",
            f"- Candidates evaluated: {details['candidates_evaluated']}",
            f"- Recommendation: {algorithm} "
            f"({details['n_clusters']} clusters)" if details["algorithm"]
            else "- Recommendation: none (abstained)",
        ]
        if details["excluded_identifiers"]:
            lines.append(
                "- Excluded ID columns: " +
                ", ".join(map(str, details["excluded_identifiers"]))
            )
        for field in ("selection_warnings", "incomplete_reasons"):
            if details[field]:
                values = "; ".join(map(str, details[field]))
                lines.append(f"- {field}: {values}")
        lines.append("\n## Candidate overview")
        view = self.ranking_table(top=top)
        if view.empty:
            lines.append("No candidates evaluated.")
        else:
            headers = ("recommended", "algorithm", "n_clusters",
                       "coverage", "silhouette", "confirmed")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")
            for _, row in view.iterrows():
                cells = [str(row.get(key, "")) for key in headers]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("\nSelection is evidence-based; lack of a "
                     "recommendation is not a failed computation.")
        return "\n".join(lines)

    def export_report(
        self, path: str | Path, *, top: int = 15
    ) -> Path:
        """Export Markdown, HTML, JSON, CSV or XLSX according to suffix.

        JSON is a human-readable summary (not a model snapshot); use
        ``save`` for trusted persistence of fitted models.
        """
        target = Path(path)
        suffix = target.suffix.lower()
        solution = self.recommended_solution()
        if suffix == ".md":
            target.write_text(self.report(top=top), encoding="utf-8")
        elif suffix == ".html":
            import html
            sections = [
                "<!doctype html><html lang='en'><meta charset='utf-8'>",
                "<title>Clustelia report</title><style>body{font:15px "
                "system-ui;max-width:1100px;margin:40px auto;color:#243042;}"
                "table{border-collapse:collapse;width:100%;font-size:13px;}"
                "th,td{border-bottom:1px solid #ddd;padding:7px;}"
                "th{text-align:left;background:#f0f3f6;}"
                "h2{margin-top:32px;}pre{white-space:pre-wrap;}</style>",
                "<h1>Clustelia experiment</h1>",
                "<h2>Evidence and decision</h2><pre>",
                html.escape(self.report(top=top).split(
                    "\n## Candidate overview")[0]), "</pre>",
                "<h2>Candidate comparison</h2>",
                self.ranking_table(top=top).to_html(index=False, escape=True),
            ]
            if solution is not None:
                sections.extend([
                    "<h2>Cluster sizes</h2>",
                    solution.sizes_table().to_html(index=False, escape=True),
                    "<h2>Feature differentiation (descriptive)</h2>",
                    solution.feature_importance_table().to_html(
                        index=False, escape=True
                    ),
                    "<h2>Cluster profiles</h2>",
                    solution.profile_table().to_html(index=False, escape=True),
                ])
            sections.append("</html>")
            target.write_text("".join(sections), encoding="utf-8")
        elif suffix == ".json":
            payload: dict[str, Any] = {
                "summary": self.summary(),
                "candidates": self.ranking_table(
                    top=None
                ).to_dict("records"),
            }
            if solution is not None:
                payload["sizes"] = solution.sizes_table().to_dict("records")
                payload["profiles"] = solution.profile_table().to_dict(
                    "records"
                )
                payload["differentiation"] = (
                    solution.feature_importance_table().to_dict("records")
                )
            def clean(value: Any) -> Any:
                """Recursively remove NaN/Inf and NumPy scalar wrappers."""
                if isinstance(value, dict):
                    return {str(k): clean(v) for k, v in value.items()}
                if isinstance(value, (list, tuple, np.ndarray)):
                    return [clean(v) for v in value]
                if isinstance(value, np.generic):
                    return clean(value.item())
                if value is None:
                    return None
                if isinstance(value, float) and not np.isfinite(value):
                    return None
                if value is pd.NA or value is pd.NaT:
                    return None
                if isinstance(value, (str, int, float, bool)):
                    return value
                return str(value)
            target.write_text(json.dumps(clean(payload), indent=2,
                                         allow_nan=False), encoding="utf-8")
        elif suffix == ".csv":
            self.ranking_table(top=None).to_csv(target, index=False)
        elif suffix == ".xlsx":
            try:
                with pd.ExcelWriter(target) as writer:
                    self.ranking_table(top=None).to_excel(
                        writer, sheet_name="Candidates", index=False
                    )
                    if solution is not None:
                        for name, frame in (
                            ("Cluster sizes", solution.sizes_table()),
                            ("Feature profiles", solution.profile_table()),
                            ("Differentiation",
                             solution.feature_importance_table()),
                            ("Distributions", solution.distribution_table()),
                        ):
                            frame.to_excel(
                                writer, sheet_name=name, index=False
                            )
            except ImportError as exc:
                raise ImportError(
                    "Install openpyxl for .xlsx export."
                ) from exc
        else:
            raise ValueError(
                "Supported reports: .md, .html, .json, .csv, .xlsx"
            )
        return target

    def plot_candidates(self, *, top: int = 12, ax: Any = None) -> Any:
        """Plot compact-objective silhouettes for fitted candidates.

        A silhouette is NOT presented as a universal performance score.
        Returns matplotlib Axes and needs ``pip install clustelia[viz]``.
        """
        if self.metadata.get("objective") != "compact":
            raise ValueError(
                "plot_candidates only supports compact objective."
            )
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install clustelia[viz] to plot.") from exc
        table = self.ranking_table(top=top)
        table = table[table["silhouette"].notna()]
        if ax is None:
            _, ax = plt.subplots()
        labels = [f"{r.algorithm} (k={r.n_clusters})"
                  for r in table.itertuples()]
        ax.barh(labels[::-1], table["silhouette"].to_numpy()[::-1])
        ax.set(xlabel="Silhouette (common reference geometry)",
               title="Candidate comparison — not a selection guarantee")
        return ax

    def candidates_table(self) -> pd.DataFrame:
        """Return every candidate record, unrounded, as a new DataFrame."""
        return pd.DataFrame(deepcopy(self._rows))

    def select(self, candidate_id: str) -> Solution:
        """Retrieve an evaluated solution without refitting it.

        Selecting a candidate manually does not make it validated.

        Raises
        ------
        KeyError
            If no successful fit exists for ``candidate_id``.
        """
        if candidate_id not in self._solutions:
            raise KeyError(f"No successful fitted candidate {candidate_id!r}.")
        return self._solutions[candidate_id]

    def compare(self, ids: Iterable[str]) -> pd.DataFrame:
        """Pairwise partition agreement on common assigned support."""
        from .evaluation import stability_pair

        ids = list(ids)
        rows = [
            {
                "left": a,
                "right": b,
                **stability_pair(
                    self.select(a).labels_, self.select(b).labels_, 1
                ),
            }
            for i, a in enumerate(ids)
            for b in ids[i + 1:]
        ]
        return pd.DataFrame(rows)

    def external_evaluate(self, y: Any) -> pd.DataFrame:
        """Post-selection ARI/NMI against external labels.

        This never changes the recommendation. ``y`` must follow the
        training row order and be complete.

        Raises
        ------
        ValueError
            If ``y`` is not a complete vector aligned with training rows.
        """
        labels_y = np.asarray(y)
        n_rows = self._training_rows()
        if (
            labels_y.ndim != 1
            or (n_rows is not None and len(labels_y) != n_rows)
            or pd.isna(labels_y).any()
        ):
            raise ValueError(
                "y must be a complete vector aligned with all training rows."
            )
        output: list[dict[str, Any]] = []
        for key, solution in self._solutions.items():
            labels = solution.labels_
            keep = labels >= 0
            evaluable = bool(keep.any())
            output.append({
                "candidate_id": key,
                "support": int(keep.sum()),
                "coverage": float(keep.mean()),
                "ari": float(adjusted_rand_score(labels_y[keep], labels[keep]))
                if evaluable else None,
                "nmi": float(
                    normalized_mutual_info_score(labels_y[keep], labels[keep])
                ) if evaluable else None,
            })
        return pd.DataFrame(output)
