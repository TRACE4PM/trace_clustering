from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import warnings
import pandas as pd
from .distance.calc_dist import levenshtein
from .trace_based_clust.clust_algos import Agglomerative_clust, DBScan_clust
from .trace_based_clust.utils import save_clusters


def clustering_algo(file_path, clustering_methode, params):
    warnings.filterwarnings('ignore')
    df = pd.read_csv(file_path, sep=";")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')

    distance_matrix = levenshtein(traces)
    result = {}

    if clustering_methode == "DBSCAN":
        clusters, cluster_assignement = DBScan_clust(distance_matrix, params)

    elif clustering_methode == "Agglomerative":
        clusters, cluster_assignement = Agglomerative_clust(distance_matrix, params)
        db_score = davies_bouldin_score(distance_matrix, cluster_assignement)
        result["Davies bouldin"] = db_score

    silhouette = silhouette_score(distance_matrix, cluster_assignement)
    result["Silhouette"] = silhouette

    save_clusters(df, cluster_assignement, traces)
    return result
