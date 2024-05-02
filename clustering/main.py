import ast
import pandas as pd
from prefixspan import PrefixSpan
import warnings
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from .distance.distance_measures import levenshtein
from .clustering_algorithms import agglomerative_clust, dbscan_clust
from .utils import silhouette_clusters, save_clusters
from .vector_based_clust.vector_based_clustering import clustering, distanceMeasures, vectorRepresentation
from .FSS_encoding.data_preparation import (frequent_subsequence_extraction, filterTraces, matrix_direct_succession,
                                            compute_fss_encoding, replace_fss_in_trace)
from .distance.utils import number_traces


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

    clusters, cluster_assignement, result = clustering(clustering_methode, distance_matrix, params)
    save_clusters(df, cluster_assignement, traces)
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
        The scores of the clustering
    """
    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(vector_representation, clustering_method, params.distance)

    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    # get the vector representation of each trace based on the choice of the user
    vectors = vectorRepresentation(vector_representation, traces)
    print(vectors)
    #
    distance_matrix = distanceMeasures(params.distance, vectors, params)

    print(distance_matrix)
    # Clustering based on the method chosen by the user
    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)
    # Saving the clusters on temp log files
    save_clusters(df, cluster_assignement, traces)
    number_traces("temp/logs/")
    return result


def feature_based_clustering(file_path):
    testTracedf = pd.read_csv(file_path)

    # df['timestamp'] = pd.to_datetime(df['timestamp'])

    # testTracedf = df.groupby("session_id")["action"].apply(list).reset_index(name='SemanticTrace')
    # testTracedf['SemanticTrace'] = testTracedf['SemanticTrace'].apply(lambda x: str(x))

    print('Number of traces is ', len(testTracedf))

    # List_traces = list(testTracedf['SemanticTrace'].apply(ast.literal_eval))

    prefixSpanRes = frequent_subsequence_extraction(testTracedf, 'SemanticTrace', min_support_percentage=99,
                                                    min_length=9)

    filteredTraces = filterTraces(testTracedf, prefixSpanRes, 'SemanticTrace')
    filteredTraces = filteredTraces[filteredTraces['hasOriginalPattern'] == 1]
    print('New Length of tracedf ', len(filteredTraces))
    # now apply FSS on filtered traces using the computed prefixSpan result of patterns
    df_activity_count, footprint_matrix = matrix_direct_succession(filteredTraces, 'SemanticTrace')
    prefixSpanRes = compute_fss_encoding(prefixSpanRes, df_activity_count, footprint_matrix)
    replaced_trace = replace_fss_in_trace(filteredTraces, 'SemanticTrace', prefixSpanRes)

    ## add clustering methodes...

    return replaced_trace
