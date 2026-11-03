"""Fuzzy C-Means search; requires the ``fuzzy`` extra."""

from sklearn.datasets import make_blobs

from clustelia import AutoCluster


def main() -> None:
    """Search FCM with a fixed fuzzifier and inspect memberships."""
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.5,
                      random_state=42)
    model = AutoCluster(objective="fuzzy", k_range=(2, 3, 4)).fit(X)
    result = model.result_
    print("Status:", result.selection_status)
    columns = ["candidate_id", "n_clusters", "fuzzy_diagnostics",
               "confirmed"]
    print(model.results_[columns].to_string(index=False))
    admissible = result.metadata["admissible_candidates"]
    print("Admissible:", admissible)
    # Selecting an alternative manually does not make it a recommendation.
    if admissible:
        solution = result.select(admissible[0])
        print("Membership semantics:", solution.membership_.semantics)
        print("First memberships:", solution.membership_.values[:3])


if __name__ == "__main__":
    main()
