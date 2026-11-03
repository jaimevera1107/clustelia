"""Deterministic candidate identifiers and per-phase seeds.

Seeds derive from stable identifiers, never from execution order, so that
reordering candidates does not change any fitted model.
"""

from __future__ import annotations

import hashlib
import json

from .config import CandidateSpec, candidate_parameters

__all__ = ["CANDIDATE_ID_LENGTH", "candidate_id", "stable_seed"]

CANDIDATE_ID_LENGTH: int = 16
_SEED_BYTES: int = 4


def stable_seed(base: int, candidate: str, phase: str) -> int:
    """Derive a 32-bit seed from a base seed, an identifier and a phase.

    Parameters
    ----------
    base : int
        User-provided ``random_state``.
    candidate : str
        Candidate identifier, or ``"common"`` for shared samples.
    phase : str
        Phase label, e.g. ``"exploration"`` or ``"subsample-3"``.

    Returns
    -------
    int
        Seed in ``[0, 2**32)``.
    """
    payload = f"{base}|{candidate}|{phase}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


def candidate_id(spec: CandidateSpec) -> str:
    """Return a stable identifier of a candidate specification."""
    payload = json.dumps(candidate_parameters(spec), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:CANDIDATE_ID_LENGTH]
