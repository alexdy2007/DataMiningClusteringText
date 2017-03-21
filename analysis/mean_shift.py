
from sklearn.cluster import MeanShift, estimate_bandwidth

bandwidth = estimate_bandwidth(pos, quantile=0.2, n_samples=47000)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(pos)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)