"""Gaussian and Bayesian mixtures selected by held-out likelihood."""

from sklearn.datasets import make_blobs

from clustelia import AutoCluster


def main() -> None:
    """Rank GMM and Bayesian GMM on the same reserved validation rows."""
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.5,
                      random_state=42)
    model = AutoCluster(
        objective="probabilistic",
        probabilistic_metric="heldout_loglikelihood",
        families=("gmm", "bgmm"),
        k_range=(2, 3, 4),
    ).fit(X)
    result = model.result_
    print("Status:", result.selection_status)
    columns = ["candidate_id", "algorithm", "n_clusters",
               "heldout_loglikelihood", "confirmed"]
    print(model.results_[columns].to_string(index=False))
    print("Fits per stage:", result.metadata["fit_breakdown"])
    if result.recommendation is not None:
        solution = result.select(result.recommendation.candidate_id)
        print("Membership semantics:", solution.membership_.semantics)


if __name__ == "__main__":
    main()
