from clustering.main import trace_based_clustering, fss_euclidean_distance,vector_based_clustering, kmeans_feature_based,feature_based_clustering
from clustering.models.cluster_params import ClusteringParams
import time
from clustering.main import drawDendrogram

# file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"
# file_path = 'C:/Users/AmaTek/Desktop/Trace clustering/notebooks/notebooks/files/lasagna.csv'
file_path = 'C:/Users/AmaTek/Desktop/Event logs/log-march_removed_cols.csv'

if __name__ == "__main__":
    start_time = time.time()

    #*******************  trace based test  *******************
    #
    # clustering_algorithm = "agglomerative"
    #
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.2, min_samples=5)
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=2, linkage='ward')
    #
    #     print(trace_based_clustering(file_path, clustering_algorithm, algorithm_params))

    # ****************** test feature based with vector representations *****************

    # clustering_algorithm = "agglomerative"
    # vector_rep = "binary representation"
    # if clustering_algorithm not in ['dbscan','kmeans', 'agglomerative', "agglomerative_euclidean"]:
    #     print("Invalid clustering algorithm.")
    # else:
    #
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.8, min_samples=9, distance='euclidean')
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=2, linkage='ward', distance='euclidean')
    #     elif clustering_algorithm == "agglomerative_euclidean":
    #         algorithm_params = ClusteringParams(nbr_clusters=3, linkage='ward', distance='euclidean')
    #
    #     print(vector_based_clustering(file_path, vector_rep,clustering_algorithm, algorithm_params))

    # ****************** test feature based FSS Encoding *****************
    #
    # result, nb = fss_euclidean_distance(file_path, 2, 90, 2)
    # print(result, nb)


    # ****************** test kmeans *****************

    algorithm_params = ClusteringParams(nbr_clusters=2)
    print(kmeans_feature_based(file_path, "binary representation", algorithm_params))

    print("temps d'execution")
    print("--- %s seconds ---" % (time.time() - start_time))