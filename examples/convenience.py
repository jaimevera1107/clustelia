"""New convenience API with a conservative identifier policy."""

import pandas as pd
from sklearn.datasets import make_blobs

from clustelia import cluster

X, _ = make_blobs(n_samples=120, centers=3, random_state=42)
frame = pd.DataFrame(X, columns=["feature_a", "feature_b"])
frame.insert(0, "record_id", [f"r{i}" for i in range(len(frame))])

result = cluster(frame, effort="quick", k_range=(2, 3, 4))
print(result.report())
print(result.ranking_table())

solution = result.recommended_solution()
if solution is not None:
    print(solution.profile_table().head())
    print(solution.sizes_table())
else:
    print("Clustelia abstained; inspect the recorded evidence.")
