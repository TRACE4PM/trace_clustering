from clustering.main import trace_based_clustering, fss_euclidean_distance,vector_based_clustering, fss_meanshift,feature_based_clustering
from clustering.models.cluster_params import ClusteringParams
import time
from clustering.main import drawDendrogram

# file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"
file_path = "/home/ania/Desktop/Code Noura/feature_based/test logs/removed_cols_Hyphen.csv"
# file_path = "/home/ania/Desktop/trace_clustering/notebooks/filtered_log_hyphen.csv"
# file_path = "/home/ania/Downloads/test.csv"
# file_path = "/home/ania/Desktop/Simulated_data/loop_actions.csv"
# file_path = "/home/ania/Desktop/Simulated_data/lasagna_actions_not_sorted.csv"



if __name__ == "__main__":
    start_time = time.time()

    #*******************  trace based test  *******************

    # clustering_algorithm = "agglomerative"
    #
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.3, min_samples=3)
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=5
    #                                             , linkage='complete')
    #
    #     print(trace_based_clustering(file_path, clustering_algorithm, algorithm_params))

    # ****************** test feature based with vector representations *****************

    # clustering_algorithm = "agglomerative"
    # vector_rep = "binary representation"
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.2, min_samples=2, distance='jaccard')
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=3, linkage='complete', distance='hamming')
    #     #hamming + complete => frequency
    #     # jaccard => binary
    #     print(vector_based_clustering(file_path, vector_rep, clustering_algorithm, algorithm_params))

    # #
    # ****************** test feature based FSS Encoding *****************


    print(fss_euclidean_distance(file_path, 2, 99, 10))

    # ****************** feature based test *****************


    # print(fss_meanshift(file_path, 'jaccard',  99, 9))

    print("temps d'execution")
    print("--- %s seconds ---" % (time.time() - start_time))