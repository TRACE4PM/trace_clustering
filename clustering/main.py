import pandas as pd
import warnings
from .clustering_methods.clustering_algorithms import clustering, meanshift
from .distance.distance_measures import distanceMeasures
from .distance.distance_measures import levenshtein
from .feature_based_clustering.vector_representation import vectorRepresentation, get_FSS_encoding
from .utils import save_clusters
from .utils import number_traces, empty_directory


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
    distance_matrix = levenshtein(traces)
    print("dist matrix", distance_matrix)

    # Clustering based on the distance matrix and the chosen method
    clusters, cluster_assignement, result = clustering(clustering_methode, distance_matrix, params)
    save_clusters(df, cluster_assignement, traces)
    nb = number_traces('temp/logs')
    return result, nb


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
    # generate the distance matrix using the distance measure the user choses
    distance_matrix = distanceMeasures(vectors, params.distance)

    print(distance_matrix)
    # Clustering based on the method chosen by the user
    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)
    # Remove the previous files in the log directory before saving the new logs
    empty_directory('temp/logs')
    # Saving the clusters on temp log files
    save_clusters(df, cluster_assignement, traces)
    nb = number_traces("temp/logs/")
    return result, nb


def feature_based_clustering(file_path, clustering_method, params):
    """
        feature based clustering using fss encoding depending on the choice of the clustering method
    """
    df = pd.read_csv(file_path, sep=";")
    traceDf = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    traceDf['trace'] = traceDf['trace'].apply(lambda x: str(x))
    fss_encoded_vectors, replaced_traces = get_FSS_encoding(traceDf, 'trace', 800, 0)
    print("replaces traces ", replaced_traces)
    columns_to_keep =['client_id', 'trace']
    # Create a new DataFrame with only the specified columns
    trace_cols= replaced_traces[columns_to_keep]
    distance_matrix = distanceMeasures(fss_encoded_vectors, params.distance)
    # TODO :
    #   fix levenshtein distances, it takes too much time (96seconds) compared to the other distance measures (1s)
    # distance_matrix = levenshtein(fss_encoded_vectors)

    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)

    # Remove the previous files in the log directory before saving the new logs
    empty_directory('temp/logs')

    return result


def fss_meanshift(file_path, params):
    """
    feature based clustering using fss encoding depending on the choice of the clustering method
    """
    df = pd.read_csv(file_path, sep=";")
    traceDf = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    traceDf['trace'] = traceDf['trace'].apply(lambda x: str(x))

    fss_encoded_vectors, replaced_traces = get_FSS_encoding(traceDf, 'trace', 30, 0)
    distance_matrix = distanceMeasures(fss_encoded_vectors, params.distance)
    columns_to_keep = ['client_id', 'trace']
    # Create a new DataFrame with only the specified columns
    trace_cols = replaced_traces[columns_to_keep]
    n_clusters_, labels_ms, result_df, scores = meanshift(distance_matrix, trace_cols)

    empty_directory('temp/logs')
    # Save traces of each cluster into separate CSV files
    for cluster_id in range(n_clusters_):
        cluster_indices = result_df[result_df['cluster_id'] == cluster_id].index
        cluster_traces = df.iloc[cluster_indices][['client_id', 'action', 'timestamp']]
        cluster_traces.to_csv(f'temp/logs/cluster_log_{cluster_id}.csv',  sep=';',index=False)

    nb = number_traces("temp/logs/")
    return scores, nb