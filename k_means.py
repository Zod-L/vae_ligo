from sklearn.cluster import KMeans
import numpy as np
import os

classes = [dir for dir in os.listdir("test")]
classes.sort()

data = []
labels = []
for i, dir in enumerate(classes):
    for fname in os.listdir(os.path.join("test", dir)):
        data.append(np.load(os.path.join("test", dir, fname)))
        labels.append(i)

data = np.vstack(data)
labels = np.array(labels)
kmeans = KMeans(n_clusters=len(classes), random_state=0, n_init="auto").fit(data)

mis = 0
for i in range(len(classes)):
    mask = (labels == i)
    cluster = kmeans.labels_[mask]
    if i == 6:
        print((cluster == 1).sum())
    cur_mis = (cluster != np.argmax(np.bincount(cluster))).sum()
    print(f"{classes[i]} : 1 - {cur_mis} / {cluster.shape[0]} = {1 - cur_mis / cluster.shape[0]}")
    mis += cur_mis
print(f"total : 1 - {mis} / {data.shape[0]} = {1 - mis / data.shape[0]}")

