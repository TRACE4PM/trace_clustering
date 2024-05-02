import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from .vector_representation import getBinaryRep, getFreqRep, extractRelativeFreq
from ..distance.distance_measures import cosine_distance, jaccard_distance, hamming_distance
from ..utils import silhouette_clusters
from ..clustering_algorithms import agglomerative_clust, dbscan_clust


def clustering(clustering_method, distance_matrix, params):
    result = {}
    if clustering_method.lower() == "dbscan":
        clusters, cluster_assignement = dbscan_clust(distance_matrix, params)

    elif clustering_method.lower() == "agglomerative":
        clusters, cluster_assignement = agglomerative_clust(distance_matrix, params)

    db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
    result["Davies bouldin"] = db_score
    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    result["Silhouette"] = silhouette
    result["Number of clusters"] = len(np.unique(cluster_assignement))
    result["Silhouette of each cluster"] = silhouette_clusters(distance_matrix, cluster_assignement)

    return clusters, cluster_assignement, result


def distanceMeasures(distance, vectors, params):
    distance_matrix = []
    if distance == "hamming":
        distance_matrix = hamming_distance(vectors)
    elif params.distance == "jaccard":
        distance_matrix = jaccard_distance(vectors)
    elif params.distance == "cosine":
        distance_matrix = cosine_distance(vectors)

    return distance_matrix


def vectorRepresentation(vector_representation, traces):
    vectors = []
    if vector_representation == "binary representation":
        vectors = getBinaryRep(traces, "client_id", "trace")
    elif vector_representation == "frequency representation":
        vectors = getFreqRep(traces, "client_id", "trace")
    elif vector_representation == "relative frequency representation":
        vectors = extractRelativeFreq(traces, "client_id", "trace")

    return vectors
