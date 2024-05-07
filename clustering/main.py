import pandas as pd
import warnings
from .clustering_methods.clustering_algorithms import clustering
from .distance.distance_measures import distanceMeasures
from .distance.distance_measures import levenshtein
from .feature_based_clustering.vector_representation import vectorRepresentation, get_FSS_encoding
from .utils import save_clusters
from .utils import number_traces


def trace_based_clustering(file_path, clustering_methode, params):
    """
    this function does the trace based clustering on a log file after generating the traces of each user
    and calculating the distance matrix between the traces
    the resulting clusters are transformed to logs  and saved in temp files

    Args:
        file_path: log file in a csv format with the columns 'client_id', 'client_id', 'timestamp'
        clustering_methode: dbscan or agglomerative clustering algorithms
        params: parameters of each clustering algorithm
            epsilon and min_samples => DBSCAN
            n_cluster and linkage criteria => Agglomerative

    Returns:
        Davis bouldin score
        Number of clusters
        Silhouette score of the clustering and for each cluster
    """

    warnings.filterwarnings('ignore')
    df = pd.read_csv(file_path, sep=";")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # generating the traces of each user by grouping their actions
    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    # calculated the normalized levenshtein distance matrix for the traces
    distance_matrix = levenshtein(traces['trace'])
    print("dist matrix", distance_matrix)

    # Clustering based on the distance matrix and the chosen method
    clusters, cluster_assignement, result = clustering(clustering_methode, distance_matrix, params)
    save_clusters(df, cluster_assignement, traces, 'trace')
    return result


def vector_based_clustering(file_path, vector_representation, clustering_method, params):
    """
    this function does the feature based clustering on the traces by representing each trace by a vector and
    and calculating the distance matrix between each vector

    Args:
        file_path: log file in a csv format with the columns 'client_id', 'client_id', 'timestamp'
        vector_representation : Binary Representation, frequency based representation, or Relative frequency
        clustering_methode: dbscan or agglomerative clustering algorithms
        params: parameters of each clustering algorithm
            epsilon and min_samples => DBSCAN
            n_cluster and linkage criteria => Agglomerative
            distance : either Jaccard, Cosine, or Hamming distance
    Returns:
        Davis bouldin score
        Number of clusters
        Silhouette score of the clustering and for each cluster
    """
    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(vector_representation, clustering_method, params.distance)
    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    # get the vector representation of each trace based on the choice of the user
    vectors = vectorRepresentation(vector_representation, traces)
    print(vectors)

    distance_matrix = distanceMeasures(vectors, params.distance)

    print(distance_matrix)
    # Clustering based on the method chosen by the user
    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)
    # Saving the clusters on temp log files
    save_clusters(df, cluster_assignement, traces)
    number_traces("temp/logs/")
    return result


def feature_based_clustering(file_path, clustering_method, params):
    """
        feature based clustering using fss encoding dependign on the choice of the clusterig method

    """
    df = pd.read_csv(file_path, sep=";")
    traceDf = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    traceDf['trace'] = traceDf['trace'].apply(lambda x: str(x))
    print(traceDf)
    fss_encoded_vectors = get_FSS_encoding(traceDf, 'trace', 30, 0)
    distance_matrix = distanceMeasures(fss_encoded_vectors, params.distance)

    # TODO :
    #   fix levenshtein distances, it takes too much time (96seconds) compared to the other distance measures (1s)
    # distance_matrix = levenshtein(fss_encoded_vectors)

    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)

    return result
