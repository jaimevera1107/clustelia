"""Automatic grid over several compact families."""

from sklearn.datasets import make_blobs

from clustelia import AutoCluster


def main() -> None:
    """Compare five compact families with the same evaluation engine."""
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.4,
                      random_state=42)
    model = AutoCluster(
        families=("kmeans", "minibatch", "birch", "bisecting", "spectral"),
        k_range=(2, 3, 4),
    ).fit(X)
    print("Status:", model.result_.selection_status)
    columns = ["candidate_id", "algorithm", "n_clusters", "silhouette",
               "confirmed"]
    print(model.results_[columns].to_string(index=False))


if __name__ == "__main__":
    main()
