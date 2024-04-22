from clustering.main import clustering_algo
from clustering.trace_based_clust.models.cluster_params import ClusteringParams

file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"


if __name__ == "__main__":
    # clustering_algorithm = input("Enter clustering algorithm (DBSCAN/Agglomerative): ").strip()
    clustering_algorithm = "agglomerative"

    if clustering_algorithm not in ['dbscan', 'agglomerative']:
        print("Invalid clustering algorithm.")
    else:

        if clustering_algorithm == 'dbscan':
            algorithm_params = ClusteringParams(eps=0.3, min_samples=5)
        elif clustering_algorithm == 'agglomerative':
            algorithm_params = ClusteringParams(n_clusters=2, linkage='single')

        print(algorithm_params.n_clusters)
        print(clustering_algo(file_path, clustering_algorithm, algorithm_params))
