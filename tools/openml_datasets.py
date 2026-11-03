"""Frozen registry of OpenML datasets for the public benchmark.

The partition into development and reserved datasets was drawn once, before
any result was seen, by a stratified random permutation (numeric-only versus
datasets with nominal features) with seed ``PARTITION_SEED``; the
structureless ``balance-scale`` grid is a negative control kept in
development. Each dataset is pinned by OpenML id, version and the md5 of
its ARFF file, which is checked on every load.

Preprocessing protocol (fixed before any result was seen):

1. Drop the target and the attributes OpenML marks as ignorable.
2. Drop columns with more than ``MAX_MISSING_FRACTION`` missing values, then
   keep complete rows only.
3. Leave the schema to clustelia's inference (numeric dtypes numeric,
   everything else nominal); no ordinal or binary declarations.
4. Subsample uniformly to ``MAX_ROWS`` rows with ``SUBSAMPLE_SEED`` when
   larger, without using the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

PARTITION_SEED: int = 20261031
MAX_MISSING_FRACTION: float = 0.20
MAX_ROWS: int = 2000
SUBSAMPLE_SEED: int = 0


@dataclass(frozen=True)
class OpenMLDataset:
    """Pinned description of one OpenML dataset.

    Attributes
    ----------
    name : str
        Benchmark name.
    data_id : int
        OpenML dataset id.
    version : int
        OpenML version of that id.
    md5 : str
        md5 checksum of the ARFF file published by OpenML.
    split : {"development", "reserved"}
        Benchmark split.
    labels_are_outcome : bool
        Whether the class is an outcome (risk, income, diagnosis) rather
        than a grouping of the features; ARI near zero is then expected.
    negative_control : bool
        Whether the dataset has no cluster structure by construction.
    """

    name: str
    data_id: int
    version: int
    md5: str
    split: str
    labels_are_outcome: bool = False
    negative_control: bool = False


REGISTRY: tuple[OpenMLDataset, ...] = (
    # Development.
    OpenMLDataset("balance-scale", 11, 1, "76938608d472f620c170cef9c8c1fa65",
                  "development", negative_control=True),
    OpenMLDataset("ecoli", 39, 1, "0b6e3ebb06a12f1b6e867c715bd1793d",
                  "development"),
    OpenMLDataset("vehicle", 54, 1, "fbba18157b188f309d772f9ca4e578f5",
                  "development"),
    OpenMLDataset("glass", 41, 1, "d38f2d5484e30ec447bbd8b6d3354460",
                  "development"),
    OpenMLDataset("segment", 36, 1, "037621812203bfd85c8dab1d7f16ebc6",
                  "development"),
    OpenMLDataset("dermatology", 35, 1, "59d9a28108aced1e9df355d36359d042",
                  "development"),
    OpenMLDataset("zoo", 62, 1, "a84cd1e10cc766cd32fd329c2ada36ac",
                  "development"),
    OpenMLDataset("credit-g", 31, 1, "9a475053fed0c26ee95cd4525e50074c",
                  "development", labels_are_outcome=True),
    OpenMLDataset("cmc", 23, 1, "3149646ecff276abac3e892d1556655f",
                  "development", labels_are_outcome=True),
    OpenMLDataset("credit-approval", 29, 1,
                  "a948be968270a823fce432227c1aa36e", "development",
                  labels_are_outcome=True),
    # Reserved: run once per methodology version.
    OpenMLDataset("yeast", 181, 1, "6408a46d338dcfb92f617ad3d2a328a0",
                  "reserved"),
    OpenMLDataset("wdbc", 1510, 1, "7aa183d3657e364911ced0cbd6b272bd",
                  "reserved"),
    OpenMLDataset("seeds", 1499, 1, "e997c7fed8ca4ba9333e89b25ba8f695",
                  "reserved"),
    OpenMLDataset("tae", 48, 1, "75b565e37610e0d8ee961c41ebf87e8b",
                  "reserved"),
    OpenMLDataset("heart-c", 49, 1, "f255fbfe272738f3da960f63a451ca0a",
                  "reserved", labels_are_outcome=True),
    OpenMLDataset("lymph", 10, 1, "f91f03d0535d532c8795787a51005caf",
                  "reserved"),
    OpenMLDataset("mushroom", 24, 1, "c8e898e453167e76d617eff67d198579",
                  "reserved"),
    OpenMLDataset("adult", 1590, 2, "bb6510925e5d4b23d136715febb2cdf5",
                  "reserved", labels_are_outcome=True),
)
"""Pinned datasets; never edit an entry after its split has been run."""

BY_NAME: dict[str, OpenMLDataset] = {d.name: d for d in REGISTRY}


def load(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Download (cached), verify and preprocess one registered dataset.

    Returns
    -------
    frame : pandas.DataFrame
        Features after the preprocessing protocol.
    labels : numpy.ndarray
        Class labels aligned with ``frame``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    ValueError
        If OpenML serves a different version or checksum than pinned.
    """
    spec = BY_NAME[name]
    bunch = fetch_openml(data_id=spec.data_id, as_frame=True,
                         parser="liac-arff")
    details = bunch.details
    if int(details["version"]) != spec.version:
        raise ValueError(f"{name}: OpenML version {details['version']} "
                         f"!= pinned {spec.version}.")
    if details.get("md5_checksum") != spec.md5:
        raise ValueError(f"{name}: md5 {details.get('md5_checksum')} "
                         f"!= pinned {spec.md5}.")
    frame = bunch.data.copy()
    labels = pd.Series(bunch.target, index=frame.index).astype(str)
    return preprocess(frame, labels)


def preprocess(
    frame: pd.DataFrame, labels: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply steps 2 to 4 of the protocol (the target is already split).

    Categorical dtypes become object strings so clustelia's inference sees
    them as nominal; numeric columns stay numeric.
    """
    missing = frame.isna().mean()
    frame = frame.loc[:, missing <= MAX_MISSING_FRACTION]
    complete = frame.notna().all(axis=1)
    frame, labels = frame.loc[complete], labels.loc[complete]
    if len(frame) > MAX_ROWS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        keep = np.sort(rng.choice(len(frame), MAX_ROWS, replace=False))
        frame, labels = frame.iloc[keep], labels.iloc[keep]
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
    return frame.reset_index(drop=True), labels.to_numpy()


def names(split: str) -> list[str]:
    """Registered dataset names of one split, in registry order."""
    return [d.name for d in REGISTRY if d.split == split]
