"""Gower geometry on a mixed numeric/nominal table with identifiers."""

import numpy as np
import pandas as pd

from clustelia import AutoCluster, GeometrySpec, Schema


def main() -> None:
    """Cluster firms by income and activity, excluding the identifier."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "firm_id": np.arange(120),
        "income": np.r_[rng.normal(100, 5, 60), rng.normal(300, 10, 60)],
        "activity": ["retail"] * 60 + ["industry"] * 60,
    })
    model = AutoCluster(
        schema=Schema(numeric=["income"], categorical=["activity"],
                      identifiers=["firm_id"]),
        geometry=GeometrySpec(kind="gower", scaling="none"),
        k_range=(2, 3, 4),
    ).fit(frame)
    result = model.result_
    print("Status:", result.selection_status)
    columns = ["candidate_id", "n_clusters", "silhouette", "confirmed"]
    print(model.results_[columns].to_string(index=False))
    if result.recommendation is not None:
        solution = result.select(result.recommendation.candidate_id)
        print("Assignment of one row:", solution.assign(frame.iloc[:1]))
        print("Profiles:", solution.describe())


if __name__ == "__main__":
    main()
