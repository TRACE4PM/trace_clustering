from clustering.trace_based_clust.algorithm import clustering_algo


file_path = "/home/ania/Desktop/trace_clustering/services/clustering/test/result_res10k.csv"


if __name__ == "__main__":
    # clustering_algorithm = input("Enter clustering algorithm (DBSCAN/Agglomerative): ").strip()
    clustering_algorithm = "Agglomerative"
    if clustering_algorithm not in ['DBSCAN', 'Agglomerative']:
        print("Invalid clustering algorithm.")
    else:
        algorithm_params = {}
        if clustering_algorithm == 'DBSCAN':
            algorithm_params['eps'] = float(input("Enter eps value for DBSCAN: "))
            algorithm_params['samples'] = int(input("Enter min_samples value for DBSCAN: "))
        elif clustering_algorithm == 'Agglomerative':
            algorithm_params['nb'] = int(input("Enter number of clusters for Agglomerative: "))

        print(clustering_algo(file_path, clustering_algorithm, algorithm_params))
