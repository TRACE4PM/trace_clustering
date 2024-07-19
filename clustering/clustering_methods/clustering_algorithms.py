from collections import Counter

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from ..utils import silhouette_clusters, silhouetteAnalysis


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


def meanshift(distance_matrix, traces_df):
    list_keys = traces_df['client_id']
    # The following bandwidth can be automatically detected using this function
    bandwidth = estimate_bandwidth(distance_matrix)
    cluster = MeanShift(bandwidth=bandwidth)
    cluster_assignement = cluster.fit(distance_matrix).labels_

    labels_unique = np.unique(cluster_assignement)
    n_clusters_ = len(labels_unique)
    print('Estimated number of clusters: %d' % n_clusters_)
    # result_df = pd.DataFrame({'client_id': list_keys, 'cluster_id': labels_ms})
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_assignement)) - (1 if -1 in cluster_assignement else 0)

    element_counts = Counter(cluster_assignement)
    unique_elements = element_counts.keys()
    occurrences = element_counts.values()
    # Print unique values and their occurrences
    for element, count in zip(unique_elements, occurrences):
        print(f"Cluster: {element}, Number of traces: {count}")
    result = {}
    # return cluster_assignement

    db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
    result["Davies bouldin"] = db_score
    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    result["Silhouette"] = silhouette
    result["Number of clusters"] = len(np.unique(cluster_assignement))
    result["Silhouette of each cluster"] = silhouette_clusters(distance_matrix, cluster_assignement)

    return n_clusters_, cluster, cluster_assignement, result


# *****************

# clustering method depends on the choice of the user

def clustering(clustering_method, data, params):
    print(clustering_method, params )
    result = {}
    data = np.array(data)
    if clustering_method.lower() == "dbscan":
        clusters, cluster_assignement = dbscan_clust(data, params)
    elif clustering_method.lower() == "agglomerative":
        clusters, cluster_assignement = agglomerative_clust(data, params)
    # # elif clustering_method.lower() == "meanshift":
    # #     clusters, cluster_assignement = meanshift(data, params)
    elif clustering_method.lower() == "agglomerative_euclidean":
        print("euclidean disttttt")
        clusters, cluster_assignement = agglomerative_euclidean(data, params)

    print(clusters, cluster_assignement)
    # Evaluating the clusters
    mask = cluster_assignement != -1
    filtered_distance_matrix = data[np.ix_(mask, mask)]
    filtered_cluster_assignment = cluster_assignement[mask]

    if len(np.unique(filtered_cluster_assignment)) > 1:  # Ensure at least two clusters for evaluation
        # Evaluating the clusters without outliers
        db_score = davies_bouldin_score(filtered_distance_matrix, filtered_cluster_assignment)
        silhouette = silhouette_score(filtered_distance_matrix, filtered_cluster_assignment)
    else:
        db_score = float('nan')  # Not enough clusters to evaluate
        silhouette = float('nan')  # Not enough clusters to evaluate

    result["Davies Bouldin"] = db_score
    result["Silhouette"] = silhouette
    result["Number of clusters"] = len(np.unique(filtered_cluster_assignment))
    result["Silhouette of each cluster"] = silhouette_clusters(filtered_distance_matrix, filtered_cluster_assignment)
    return clusters, cluster_assignement, result


#
# **********************

def agglomerative_euclidean(data, n_clusters):
    # Perform agglomerative clustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric='euclidean')
    cluster_assignments = cluster.fit_predict(data)

    # Calculate silhouette score
    silhouette = silhouette_score(X=data, labels=cluster_assignments, metric='euclidean')
    # Calculate Davies-Bouldin score
    davies_bouldin = davies_bouldin_score(data, cluster_assignments)
    # Perform silhouette analysis
    silhouette_analysis = silhouetteAnalysis(data, cluster_assignments, n_clusters, 'euclidean')

    result = {
        "Davies-Bouldin": davies_bouldin,
        "Silhouette Score": silhouette,
        "Silhouette Analysis": silhouette_analysis,
    }

    return cluster,cluster_assignments, result


def applyHAC(linkage, num_of_clusters, metric, data):
    agglomerative = AgglomerativeClustering(n_clusters=num_of_clusters, metric=metric, linkage=linkage)
    # Fit the model
    cluster_labels = agglomerative.fit_predict(data)
    print(f"The silhouette score for {num_of_clusters} clusters using {linkage} linkage is: ",
          silhouette_score(X=data, labels=agglomerative.labels_, metric=metric))

    print(f"The davies bouldin score for {num_of_clusters} clusters using {linkage} linkage is: ",
          davies_bouldin_score(data, agglomerative.labels_))
    # Count occurrences of each element in the list
    # if this line poses an error use instead the following commented lines
    element_counts = Counter(cluster_labels)
    # element_counts = Counter(list(cluster_labels_)
    # unique_elements = list(element_counts.keys())
    # occurrences = list(element_counts.values())

    # Get unique values and their occurrences
    unique_elements = element_counts.keys()
    occurrences = element_counts.values()
    # Print unique values and their occurrences
    for element, count in zip(unique_elements, occurrences):
        print(f"Cluster: {element}, Number of traces: {count}")

    return cluster_labels
