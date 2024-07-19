import numpy as np
import pandas as pd
import warnings
from .clustering_methods.clustering_algorithms import clustering, meanshift, applyHAC, agglomerative_euclidean
from .distance.distance_measures import distanceMeasures
from .distance.distance_measures import levenshtein
from .feature_based_clustering.vector_representation import vectorRepresentation, get_FSS_encoding
from .utils import save_clusters, save_clusters_fss, drawDendrogram
from .utils import number_traces, silhouetteAnalysis

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
    df = df.sort_values(by='timestamp', ascending=True)
    # generating the traces of each user by grouping their actions
    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    # calculated the normalized levenshtein distance matrix for the traces
    distance_matrix = levenshtein(traces['trace'].array)
    print("dist matrix", distance_matrix)

    # generating the dendrogram for hieararchical clustering
    # drawDendrogram(distance_matrix, params.linkage)

    # Clustering based on the distance matrix and the chosen method
    clusters, cluster_assignement, result = clustering(clustering_methode, distance_matrix, params)
    print(clusters, cluster_assignement)
    # Remove the previous files in the log directory before saving the new logs
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

    df = df.sort_values(by='timestamp', ascending=True)
    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    # get the vector representation of each trace based on the choice of the user
    vectors = vectorRepresentation(vector_representation, traces)
    distance_matrix = distanceMeasures(vectors, params.distance)
    print(distance_matrix)
    # drawDendrogram(distance_matrix, params.linkage)
    # Clustering based on the method chosen by the user
    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)
    # Saving the clusters on temp log files
    save_clusters(df, cluster_assignement, traces)
    nb = number_traces("temp/logs/")
    return result, nb


def feature_based_clustering(file_path, clustering_method, params, min_support, min_length):
    """
        feature based clustering using fss encoding depending on the choice of the clustering method
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
    df = df.sort_values(by='timestamp', ascending=True)
    trace_df = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
    trace_df['trace'] = trace_df['trace'].apply(lambda x: str(x))
    fss_encoded_vectors, replaced_traces = get_FSS_encoding(trace_df, 'trace', min_support, min_length)
    print("replaces traces ", replaced_traces)
    columns_to_keep =['client_id', 'trace']

    distance_matrix = distanceMeasures(fss_encoded_vectors, params.distance)

    # drawDendrogram(distance_matrix, params.linkage )
    clusters, cluster_assignement, result = clustering(clustering_method, distance_matrix, params)

    labels_unique = np.unique(cluster_assignement)
    nbr_clusters = len(labels_unique)
    # Create a new DataFrame with only the specified columns
    columns_to_keep = ['client_id', 'trace']
    trace_cols = replaced_traces[columns_to_keep]
    list_clients = trace_cols['client_id']  # get a list of all the client ids
    result_df = pd.DataFrame({'client_id': list_clients, 'cluster_id': cluster_assignement})

    # Save traces of each cluster into separate CSV files
    save_clusters_fss(nbr_clusters, df, result_df)
    # get the number of traces in each cluster
    nb = number_traces("temp/logs/")

    return result, nb

def fss_euclidean_distance(file_path, n_cluster, min_support, min_length):
    """
        feature based clustering using fss encoding and without a distance measure
        using Agglomerative clustering with Ward linkage and Euclidean distance
     Args:
        file_path: log file in a csv format with the columns 'client_id', 'client_id', 'timestamp'
        vector_representation : Binary Representation, frequency based representation, or Relative frequency
        clustering_methode: dbscan or agglomerative clustering algorithms
        n_cluster and linkage criteria => Agglomerative
    Returns:
        Davis bouldin score
        Number of clusters
        Silhouette score of the clustering and for each cluster
    """
    df = pd.read_csv(file_path, sep=";")
    df = df.sort_values(by='timestamp', ascending=True)
    print(df.head())
    trace_df = df.groupby("client_id")["action"].apply(list).reset_index(name='SemanticTrace')
    print(trace_df.head())
    trace_df['SemanticTrace'] = trace_df['SemanticTrace'].apply(lambda x: str(x))

    replaced_traces = get_FSS_encoding(trace_df, 'SemanticTrace', min_support, min_length)

    # Convert the list of vectors to a numpy array
    data =list(replaced_traces['SemanticTrace_FSSEncoded_Padded'])

    # generating the dendrogram based on the linkage criteria
    # drawDendrogram(data, 'ward', 'Using Improved FSS on all traces')

    clusters, cluster_assignement, result = agglomerative_euclidean(data, n_cluster)
    # replaced_traces['Cluster_Labels'] = fss_cluster_labels_hac
    # cluster, cluster_assignement, result = clustering("agglomerative_ward",data, nbr_clusters)
    #
    columns_to_keep = ['client_id', 'SemanticTrace']
    trace_cols = replaced_traces[columns_to_keep]

    list_clients = trace_cols['client_id']  #get a list of all the client ids
    result_df = pd.DataFrame({'client_id': list_clients, 'cluster_id': cluster_assignement})
    # Save traces of each cluster into separate CSV files
    save_clusters_fss(n_cluster,df, result_df)
    nb = number_traces("temp/logs/")  # get the number of traces in each cluster
    return result, nb

def fss_meanshift(file_path, distance, min_support, min_length):
    """
        feature based clustering using FSS encoding, we choose the distance measure and the
     clustering algorithm is Meanshift
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
    df = df.sort_values(by='timestamp', ascending=True)
    trace_df = df.groupby("client_id")["action"].apply(list).reset_index(name='SemanticTrace')
    trace_df['SemanticTrace'] = trace_df['SemanticTrace'].apply(lambda x: str(x))

    replaced_traces = get_FSS_encoding(trace_df, 'SemanticTrace', min_support, min_length)

    # Convert the list of vectors to a numpy array
    data = list(replaced_traces['SemanticTrace_FSSEncoded_Padded'])

    # agglomerative clustering using ward linkage and euclidean distance
    print("replaced ", replaced_traces)
    distance_matrix = distanceMeasures(data, distance)

    # Create a new DataFrame with only the specified columns
    columns_to_keep = ['client_id', 'SemanticTrace']
    trace_cols = replaced_traces[columns_to_keep]
    nbr_clusters, cluster, cluster_assignement, result = meanshift(distance_matrix, trace_cols)

    list_clients = trace_cols['client_id']  # get a list of all the client ids
    result_df = pd.DataFrame({'client_id': list_clients, 'cluster_id': cluster_assignement})

    silhouetteAnalysis(data, cluster_assignement, nbr_clusters)

    # Save traces of each cluster into separate CSV files
    save_clusters_fss(nbr_clusters, df, result_df)
    # get the number of traces in each cluster
    # nb = number_traces("temp/logs/")
    return result


