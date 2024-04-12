from sklearn.cluster import KMeans


def kmeans_clust(best_k, distance_matrix):
    kmeans = KMeans(n_clusters=best_k)
    X = distance_matrix.reshape(-1, 1)
    clusters = kmeans.fit_predict(X)

    # Assigning clusters to client_ids
    cluster_assignments = {i: [] for i in range(best_k)}
    for i, cluster_id in enumerate(clusters):
        client_id1, client_id2, _ = distances[i]
        cluster_assignments[cluster_id].append(client_id1)
        cluster_assignments[cluster_id].append(client_id2)

    return cluster_assignments
