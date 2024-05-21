from clustering.main import trace_based_clustering, fss_euclidean_distance,vector_based_clustering, fss_meanshift,feature_based_clustering
from clustering.models.cluster_params import ClusteringParams
import time

# file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"
file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/lasagna.csv"
# file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/simulated_data.csv"

if __name__ == "__main__":
    start_time = time.time()

    #*******************  trace based test  *******************

    # clustering_algorithm = "agglomerative"
    #
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.3, min_samples=5)
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=3, linkage='single')

    #     print(trace_based_clustering(file_path, clustering_algorithm, algorithm_params))

    # ****************** test feature based with vector representations *****************

    # clustering_algorithm = "dbscan"
    # vector_rep = "binary representation"
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(epsilon=0.2, min_samples=2, distance = 'jaccard')
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(nbr_clusters=3, linkage='single', distance = 'cosine')
    #
    #     print(vector_based_clustering(file_path,vector_rep, clustering_algorithm, algorithm_params))


    # ****************** feature based test *****************

    #
    clustering_algorithm = "agglomerative"
    if clustering_algorithm not in ['dbscan', 'agglomerative']:
        print("Invalid clustering algorithm.")
    else:

        if clustering_algorithm == 'dbscan':
            algorithm_params = ClusteringParams(epsilon=0.5, min_samples=2, distance = 'jaccard')
        elif clustering_algorithm == 'agglomerative':
            algorithm_params = ClusteringParams(nbr_clusters=3, linkage='complete', distance = 'hamming')

    print(fss_euclidean_distance(file_path, 3, 80, 0))

    print("temps FSS + clustering")
    print("--- %s seconds ---" % (time.time() - start_time))