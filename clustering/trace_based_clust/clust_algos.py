from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


def dbscan_clust(distance_matrix, params):
    cluster = DBSCAN(eps=params.eps, min_samples=params.min_samples, metric='precomputed')
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments


def agglomerative_clust(distance_matrix, params):
    cluster = AgglomerativeClustering(n_clusters=params.n_clusters, linkage=params.linkage)
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments

