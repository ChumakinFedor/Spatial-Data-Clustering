from sklearn.datasets import make_blobs
import csv

X, y = make_blobs(
    n_samples=300,
    n_features=2,
    centers=3,
    cluster_std=0.8,
    random_state=50 # seed можно менять для генерации разных данных
)

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "original_label"])
    for (x, yv), label in zip(X, y):
        writer.writerow([f"{x:.10f}", f"{yv:.10f}", int(label)])
