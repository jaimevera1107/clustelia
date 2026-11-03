"""Clustelia: explicit geometry, reproducible search and stability evidence.

The public API is re-exported here; every other module is an
implementation detail whose contents may change between releases.
"""

import logging

from ._version import __version__
from .adapters import Membership, adapter_capabilities
from .config import (
    CandidateSpec,
    GeometrySpec,
    Schema,
    SearchBudget,
    SelectionPolicy,
)
from .convenience import cluster, detect_identifiers
from .exceptions import (
    CapabilityError,
    ClusteliaError,
    MemoryBudgetError,
    NoRecommendationError,
    SearchFailedError,
)
from .persistence import load, save
from .results import ExperimentResult, Recommendation, Solution
from .search import AutoCluster

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AutoCluster",
    "CandidateSpec",
    "CapabilityError",
    "ClusteliaError",
    "ExperimentResult",
    "GeometrySpec",
    "Membership",
    "MemoryBudgetError",
    "NoRecommendationError",
    "Recommendation",
    "Schema",
    "SearchBudget",
    "SearchFailedError",
    "SelectionPolicy",
    "Solution",
    "__version__",
    "adapter_capabilities",
    "cluster",
    "detect_identifiers",
    "load",
    "save",
]
