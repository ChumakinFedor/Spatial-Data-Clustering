import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
results = pd.read_csv("results.csv")

if data.shape[0] != results.shape[0]:
    raise ValueError("Row count mismatch between data.csv and results.csv")

label_col = data.columns[2]
true_labels = data[label_col]
pred_labels = results["pred_label"]

centroids = (
    results.assign(x=results["x"], y=results["y"])
    .groupby("pred_label", as_index=False)[["x", "y"]]
    .mean()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(data["x"], data["y"], c=true_labels, cmap="viridis", s=25)
axes[0].set_title("True Labels")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].scatter(results["x"], results["y"], c=pred_labels, cmap="viridis", s=25)
axes[1].scatter(
    centroids["x"],
    centroids["y"],
    c="red",
    marker="X",
    s=180,
    edgecolors="black",
    linewidths=1.0
)
axes[1].set_title("Predicted Labels")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
