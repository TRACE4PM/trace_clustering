import pandas as pd
import asyncio
from distance.calc_dist import levenshtein
from trace_based_clust.algorithm import DBScan_clust, Agglomerative_clust
from sklearn.metrics import silhouette_score
from trace_based_clust.utils import save_clusters
from sklearn.metrics import davies_bouldin_score
import numpy as np
import warnings

file_path = "/services/clustering/test/result_res10k.csv"


def clusetering_algo(file_path, clustering_methode, params):

    warnings.filterwarnings('ignore')

    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    distance_matrix = levenshtein(traces)
    print("Distance matrix")
    print(distance_matrix)

    if clustering_methode =="DBSCAN":
        clusters, cluster_assignement = DBScan_clust(distance_matrix, params)

    elif clustering_methode == "Agglomerative":
        clusters, cluster_assignement = Agglomerative_clust(distance_matrix, params)

    print("number of clusters after clustering :", len(np.unique(cluster_assignement)))
    # Evaluating clustering results
    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    print("silhouette score :", silhouette)

    db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
    print("davies bouldin score : ", db_score)

    save_clusters(df, cluster_assignement, traces)


if __name__ == "__main__":
    # clustering_algorithm = input("Enter clustering algorithm (DBSCAN/Agglomerative): ").strip()
    clustering_algorithm = "DBSCAN"
    if clustering_algorithm not in ['DBSCAN', 'Agglomerative']:
        print("Invalid clustering algorithm.")
    else:
        algorithm_params = {}
        if clustering_algorithm == 'DBSCAN':
            algorithm_params['eps'] = float(input("Enter eps value for DBSCAN: "))
            algorithm_params['samples'] = int(input("Enter min_samples value for DBSCAN: "))
        elif clustering_algorithm == 'Agglomerative':
            algorithm_params['nb'] = int(input("Enter number of clusters for Agglomerative: "))

        clusetering_algo(clustering_algorithm, algorithm_params)