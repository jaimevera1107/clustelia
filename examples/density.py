"""Density search; recommending requires the ``density`` extra (DBCV)."""

from sklearn.datasets import make_moons

from clustelia import AutoCluster, GeometrySpec


def main() -> None:
    """Explore DBSCAN, OPTICS and HDBSCAN on two interleaved moons."""
    X, _ = make_moons(n_samples=240, noise=0.04, random_state=42)
    model = AutoCluster(
        objective="density", geometry=GeometrySpec(scaling="none")
    ).fit(X)
    result = model.result_
    print("Status:", result.selection_status)
    columns = ["candidate_id", "algorithm", "n_clusters", "coverage", "dbcv",
               "confirmed"]
    print(result.candidates_table()[columns].to_string(index=False))
    print("Incomplete reasons:", result.metadata["incomplete_reasons"])
    print("Admissible:", result.metadata["admissible_candidates"])
    if result.recommendation is not None:
        print("Recommendation:", result.recommendation.candidate_id)


if __name__ == "__main__":
    main()
