import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering,DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import davies_bouldin_score, silhouette_score
from ..utils import silhouette_clusters

def dbscan_clust(distance_matrix, params):
    cluster = DBSCAN(eps=params.epsilon, min_samples=params.min_samples, metric='precomputed')
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments


def agglomerative_clust(distance_matrix, params):
    cluster = AgglomerativeClustering(n_clusters=params.nbr_clusters, linkage=params.linkage)
    cluster_assignments = cluster.fit_predict(distance_matrix)
    return cluster, cluster_assignments


def kmeans_clust(best_k, distance_matrix):
    kmeans = KMeans(n_clusters=best_k)
    X = distance_matrix.reshape(-1, 1)
    clusters = kmeans.fit_predict(X)
    return clusters


# *****************

# clustering method depends on the choice of the user

def clustering(clustering_method, distance_matrix, params):
    result = {}
    if clustering_method.lower() == "dbscan":
        clusters, cluster_assignement = dbscan_clust(distance_matrix, params)

    elif clustering_method.lower() == "agglomerative":
        clusters, cluster_assignement = agglomerative_clust(distance_matrix, params)

    elif clustering_method.lower() == "agglomerative_ward":
        clusters, cluster_assignement = agglomerative_ward(distance_matrix, params)

    db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
    result["Davies bouldin"] = db_score
    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    result["Silhouette"] = silhouette
    result["Number of clusters"] = len(np.unique(cluster_assignement))
    result["Silhouette of each cluster"] = silhouette_clusters(distance_matrix, cluster_assignement)

    return clusters, cluster_assignement, result

#
# **********************

# this function is for FSS encoding
# Updating it later with other methods

def meanshift(distance_matrix, traces_df):
    list_keys = traces_df['client_id']
    # The following bandwidth can be automatically detected using this function
    bandwidth = estimate_bandwidth(distance_matrix)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(distance_matrix)
    labels_ms = ms.labels_
    labels_unique = np.unique(labels_ms)
    n_clusters_ = len(labels_unique)
    print('Estimated number of clusters: %d' % n_clusters_)
    result_df = pd.DataFrame({'client_id': list_keys, 'cluster_id': labels_ms})
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels_ms)) - (1 if -1 in labels_ms else 0)
    for i in range(n_clusters_):
        firstclasse = list(labels_ms).count(i)
        print('Estimated number of', i, ' points: %d' % firstclasse)

    scores = {}
    db_score = davies_bouldin_score(distance_matrix, labels_ms)
    scores["Davies bouldin"] = db_score
    silhouette = silhouette_score(distance_matrix, labels_ms)
    scores["Silhouette"] = silhouette
    scores["Number of clusters"] = len(np.unique(labels_ms))
    scores["Silhouette of each cluster"] = silhouette_clusters(distance_matrix, labels_ms)

    return n_clusters_, labels_ms, result_df, scores

def agglomerative_ward(data, nbr_clusters):
    cluster = AgglomerativeClustering(n_clusters=nbr_clusters, linkage='ward', metric='euclidean')
    cluster_assignments = cluster.fit_predict(data)
    return cluster, cluster_assignments
