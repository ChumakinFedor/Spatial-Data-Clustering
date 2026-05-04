from sklearn.datasets import make_blobs
import csv

# Генерация синтетических данных с помощью make_blobs
X, y = make_blobs(
    n_samples=300, # количество точек
    n_features=2, # количество признаков (размерность пространства)
    centers=3, # количество кластеров
    cluster_std=0.8, # стандартное отклонение кластеров (чем меньше, тем плотнее кластеры)
    random_state=50 # seed можно менять для генерации разных данных
)

# Сохранение данных в CSV файл
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "original_label"])
    for (x, yv), label in zip(X, y):
        writer.writerow([f"{x:.10f}", f"{yv:.10f}", int(label)])