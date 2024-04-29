from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import warnings
import pandas as pd
from .distance.calc_dist import levenshtein
from .clustering_algorithms import agglomerative_clust, dbscan_clust
from .utils import silhouette_clusters, save_clusters
from .vector_based_clust.vector_based_clustering import clustering, distanceMeasures, vectorRepresentation
from .FSS_encoding.data_preparation import (frequent_subsequence_extraction, filterTraces,matrix_direct_succession,
                                            compute_fss_encoding, replace_fss_in_trace)
import ast
from prefixspan import PrefixSpan

def trace_based_clustering(file_path, clustering_methode, params):
    warnings.filterwarnings('ignore')
    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    distance_matrix = levenshtein(traces)
    result = {}

    if clustering_methode.lower() == "dbscan":
        clusters, cluster_assignement = dbscan_clust(distance_matrix, params)

    elif clustering_methode.lower() == "agglomerative":
        clusters, cluster_assignement = agglomerative_clust(distance_matrix, params)
        db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
        result["Davies bouldin"] = db_score

    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    result["Silhouette"] = silhouette

    result["Silhouette of each cluster"] = silhouette_clusters(distance_matrix, cluster_assignement)

    save_clusters(df, cluster_assignement, traces)
    return result


def vector_based_clustering(file_path, vector_representation, clustering_method, params):
    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(vector_representation, clustering_method, params.distance)

    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    vectors = vectorRepresentation(vector_representation, traces)
    print(vectors)

    distance_matrix = distanceMeasures(params.distance, vectors, params)

    print(distance_matrix)

    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)

    save_clusters(df, cluster_assignement, traces)
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
    print(prefixSpanRes)

    filteredTraces = filterTraces(testTracedf, prefixSpanRes, 'SemanticTrace')
    filteredTraces = filteredTraces[filteredTraces['hasOriginalPattern'] == 1]
    print('New Length of tracedf ', len(filteredTraces))
    # now apply FSS on filtered traces using the computed prefixSpan result of patterns
    df_activity_count, footprint_matrix = matrix_direct_succession(filteredTraces, 'SemanticTrace')
    prefixSpanRes = compute_fss_encoding(prefixSpanRes, df_activity_count, footprint_matrix)
    replaced_trace = replace_fss_in_trace(filteredTraces, 'SemanticTrace', prefixSpanRes)

    return replaced_trace
