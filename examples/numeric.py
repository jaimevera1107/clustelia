"""Automatic compact search on numeric data."""

from sklearn.datasets import make_blobs

from clustelia import AutoCluster


def main() -> None:
    """Search k in 2..5 and report the recommendation, if any."""
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.3,
                      random_state=42)
    result = AutoCluster(k_range=(2, 3, 4, 5), random_state=42).fit(X).result_
    print("Status:", result.selection_status)
    print("Fits used:", result.metadata["fits_used"])
    columns = ["candidate_id", "algorithm", "n_clusters", "silhouette",
               "confirmed"]
    print(result.candidates_table()[columns].to_string(index=False))
    if result.recommendation is not None:
        solution = result.select(result.recommendation.candidate_id)
        print("Recommendation:", solution.candidate_id)
        print("Profiles:", solution.describe())
        # External labels are used only after selection.
        print(result.external_evaluate(y).to_string(index=False))


if __name__ == "__main__":
    main()
