from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


def DBScan_clust(distance_matrix, params):
    cluster = DBSCAN(eps=params['eps'], min_samples=params['samples'], metric='precomputed')
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments


def Agglomerative_clust(distance_matrix, params):
    cluster = AgglomerativeClustering(n_clusters=params['nb'], linkage='single')
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments

