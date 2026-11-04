"""Conservative, accessible front door for the expert clustering engine.

The advanced :class:`AutoCluster` API retains its original defaults. This
module adds opt-in ergonomics without weakening validation or silently
changing model selection in the existing API.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Literal

import pandas as pd

from .config import Schema, SearchBudget, SelectionPolicy
from .results import ExperimentResult
from .search import AutoCluster

Effort = Literal["quick", "balanced", "thorough"]

_IDENTIFIER_NAME = re.compile(
    r"^(?:id|uuid|guid|identifier|record_id|row_id|entity_id)$|"
    r"(?:^|[_\-])(?:id|uuid|guid|identifier)(?:$|[_\-])",
    re.IGNORECASE,
)


def detect_identifiers(X: pd.DataFrame) -> tuple:
    """Identify unique complete columns *explicitly named* as IDs.

    A unique numeric measurement, high-cardinality category, or row index
    alone is NEVER enough evidence to exclude a feature. A likely ID whose
    values repeat is not removed; use an explicit ``Schema`` to specify
    composite or otherwise meaningful identifiers.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("detect_identifiers expects a pandas DataFrame.")
    if not X.columns.is_unique:
        raise ValueError("DataFrame columns must be unique.")
    return tuple(
        c for c in X.columns
        if isinstance(c, str)
        and _IDENTIFIER_NAME.search(c)
        and X[c].notna().all()
        and X[c].is_unique
    )


def _inferred_schema(X: pd.DataFrame, identifiers: tuple) -> Schema:
    """Resolve roles like the core validator, but exclude inferred IDs."""
    features = X.drop(columns=list(identifiers))
    numeric = tuple(features.select_dtypes(include="number").columns)
    categorical = tuple(c for c in features if c not in numeric)
    return Schema(numeric=numeric, categorical=categorical,
                  identifiers=identifiers)


def _preset(effort: Effort) -> tuple[SelectionPolicy, SearchBudget]:
    if effort == "quick":
        # Faster but substantially weaker null evidence than balanced.
        return (
            SelectionPolicy(version="P2-quick", null_replicates=2,
                            subsamples=3, seed_repeats=1,
                            shortlist_size=3),
            SearchBudget(max_fits=100, max_candidates=21,
                         timeout_s=120, metric_sample_size=500),
        )
    if effort == "balanced":
        # Bit-for-bit identical policy and budget to expert defaults.
        return SelectionPolicy(), SearchBudget()
    if effort == "thorough":
        return (
            SelectionPolicy(version="P2-thorough", null_replicates=19,
                            subsamples=15, seed_repeats=5,
                            shortlist_size=7),
            SearchBudget(max_fits=800, max_candidates=75,
                         timeout_s=1800, metric_sample_size=1500),
        )
    raise ValueError("effort must be 'quick', 'balanced', or 'thorough'.")


def cluster(
    X: Any,
    *,
    effort: Effort = "balanced",
    identifiers: Literal["auto"] | Iterable[Any] | None = "auto",
    schema: Schema | None = None,
    selection: SelectionPolicy | None = None,
    budget: SearchBudget | None = None,
    null_replicates: int | None = None,
    **kwargs: Any,
) -> ExperimentResult:
    """Run clustering and return a reportable experiment.

    Parameters
    ----------
    X : array-like or DataFrame
        Observations by columns; never modified.
    effort : {"quick", "balanced", "thorough"}
        Fixed, documented convenience presets. ``quick`` trades weaker
        null evidence for speed; ``thorough`` needs larger fit budgets.
    identifiers : "auto", iterable or None
        ``"auto"`` excludes only complete unique columns with explicit
        ID-like names (DataFrames only). An iterable declares identifiers;
        ``None`` disables inference. Explicit ``schema`` takes precedence.
    schema, selection, budget
        Expert overrides.
    null_replicates : int or None
        Explicit Monte Carlo replication depth independent from ``effort``.
        Zero disables null testing (no structural validation); >=2 enables
        the gap rule when the scope is the full grid. A small budget may
        interrupt validation; no automatic threshold weakening occurs.
        Passing both ``schema`` and non-default
        ``identifiers`` is prohibited to avoid contradictory roles.
    **kwargs
        Forwarded to :class:`AutoCluster`. The advanced API remains fully
        available, e.g. objective, geometry, families, and random_state.

    Returns
    -------
    ExperimentResult
        Results even when selection abstains; call ``summary()``,
        ``ranking_table()``, ``report()`` or ``recommended_solution()``.
        No unsupported ``labels_`` property is invented on abstention.
    """
    preset_policy, preset_budget = _preset(effort)
    if null_replicates is not None:
        if (isinstance(null_replicates, bool) or
                not isinstance(null_replicates, int)):
            raise TypeError("null_replicates must be an integer >= 0")
        if selection is not None:
            raise ValueError(
                "null_replicates conflicts with explicit selection"
            )
        preset_policy = replace(preset_policy, null_replicates=null_replicates)
    if schema is not None:
        if not (isinstance(identifiers, str) and identifiers == "auto"):
            raise ValueError("Provide either schema or identifiers, not both.")
    elif isinstance(X, pd.DataFrame):
        if isinstance(identifiers, str) and identifiers == "auto":
            excluded = detect_identifiers(X)
        elif identifiers is None:
            excluded = ()
        else:
            if isinstance(identifiers, (str, bytes)):
                raise ValueError(
                    "identifiers must be 'auto', None or columns."
                )
            excluded = tuple(identifiers)
        if excluded:
            schema = _inferred_schema(X, excluded)
    elif identifiers is not None and not (
        isinstance(identifiers, str) and identifiers == "auto"
    ):
        raise TypeError("Explicit identifier names require a DataFrame.")
    if "selection" in kwargs or "budget" in kwargs or "schema" in kwargs:
        raise TypeError(
            "Pass schema, selection and budget as named arguments."
        )
    estimator = AutoCluster(
        schema=schema,
        selection=selection or preset_policy,
        budget=budget or preset_budget,
        **kwargs,
    ).fit(X)
    result = estimator.result_
    result.metadata["effort"] = effort
    result.metadata["null_replicates_override"] = null_replicates
    result.metadata["convenience_api"] = True
    return result
