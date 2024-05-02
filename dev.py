from clustering.main import trace_based_clustering, vector_based_clustering, feature_based_clustering
from clustering.models.cluster_params import ClusteringParams
from clustering.distance.utils import number_traces

file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"
# file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/TraceTestFile.csv"

if __name__ == "__main__":

    #*******************  trace based test  *******************

    clustering_algorithm = "dbscan"

    if clustering_algorithm not in ['dbscan', 'agglomerative']:
        print("Invalid clustering algorithm.")
    else:

        if clustering_algorithm == 'dbscan':
            algorithm_params = ClusteringParams(eps=0.3, min_samples=5)
        elif clustering_algorithm == 'agglomerative':
            algorithm_params = ClusteringParams(n_clusters=2, linkage='single')

        print(algorithm_params.n_clusters)
        trace_based_clustering(file_path, clustering_algorithm, algorithm_params)


    # ****************** vector based test *****************

    # clustering_algorithm = "agglomerative"
    # vector_rep = "binary representation"
    # if clustering_algorithm not in ['dbscan', 'agglomerative']:
    #     print("Invalid clustering algorithm.")
    # else:
    #
    #     if clustering_algorithm == 'dbscan':
    #         algorithm_params = ClusteringParams(eps=0.5, min_samples=2, distance = 'jaccard')
    #     elif clustering_algorithm == 'agglomerative':
    #         algorithm_params = ClusteringParams(n_clusters=2, linkage='single', distance = 'jaccard')
    #
    #     print(vector_based_clustering(file_path, vector_rep, clustering_algorithm, algorithm_params))

        # number_traces("/home/ania/Desktop/trace_clustering/services/clustering/temp/logs")

    # ****************** feature based test *****************


    # result = feature_based_clustering(file_path)
    #
    # print("replaced traces :", result)