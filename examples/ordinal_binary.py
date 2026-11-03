"""Weighted Gower with nominal, ordinal and asymmetric binary variables."""

import numpy as np
import pandas as pd

from clustelia import AutoCluster, GeometrySpec, Schema

EDUCATION = ("primary", "secondary", "tertiary")


def main() -> None:
    """Cluster synthetic workers described by mixed variables."""
    rng = np.random.default_rng(7)
    groups = np.repeat([0, 1, 2], 80)
    frame = pd.DataFrame({
        "worker_id": np.arange(len(groups)),
        "hours": rng.normal(20 + 10 * groups, 3),
        "sector": np.array(["farming", "retail", "services"])[groups],
        "education": np.array(EDUCATION)[groups],
        "formal_contract": np.array([0, 0, 1])[groups],
    })
    schema = Schema(
        numeric=["hours"],
        categorical=["sector"],
        ordinal={"education": EDUCATION},
        binary=["formal_contract"],
        identifiers=["worker_id"],
    )
    model = AutoCluster(
        schema=schema,
        geometry=GeometrySpec("gower", "none", weights={"hours": 2.0}),
        k_range=(2, 3, 4, 5),
        random_state=7,
    ).fit(frame)
    result = model.result_
    print("Status:", result.selection_status)
    if result.recommendation is None:
        print(result.metadata["incomplete_reasons"])
        return
    solution = result.select(result.recommendation.candidate_id)
    print("Algorithm:", solution.pipeline_.spec.algorithm,
          "k =", solution.pipeline_.spec.k)
    for profile in solution.describe():
        print(profile["label"], profile["size"],
              profile["ordinal"]["education"]["median"],
              profile["binary"]["formal_contract"])
    print("New row:", model.predict(frame.iloc[:1]))


if __name__ == "__main__":
    main()
