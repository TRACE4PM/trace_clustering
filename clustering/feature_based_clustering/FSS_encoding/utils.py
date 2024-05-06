import numpy as np
import pandas as pd
from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import davies_bouldin_score, silhouette_score


def same_length_vectors(list_values_float, max):
    cpt =0
    for elt in list_values_float:
        cpt = cpt +1
        for i in range(max):
            if len(elt) != max:
                #print(len(elt))
                reste = max - len(elt)
                #print(reste)
                for j in range(reste):
                    elt.append(0)
    return(list_values_float)



def levenshtein(traces):
    """
    Levenshtein distance measure is used for trace based clustering to calculate the distance between 2 traces
    Return : Normalized Distance matrix

    """
    matrix_size = len(traces)
    distance_matrix = np.empty((matrix_size, matrix_size), float)
    normalized_levenshtein = NormalizedLevenshtein()
    j = 0
    for i in range(0, matrix_size):
        if (i == j):
            distance_matrix[i][j] = 0
        else:
            while (j < i):
                string1 = traces.iloc[i]
                string2 = traces.iloc[j]
                lev_dist = normalized_levenshtein.distance(string1, string2)
                distance_matrix[i][j] = distance_matrix[j][i] = lev_dist
                j = j + 1
            distance_matrix[i][j] = 0
            j = 0
    return distance_matrix



def convertLogs(logFilePath, outputFilePath):
    logs = pd.read_csv(logFilePath, delimiter=';')
    groupedlogs = logs.groupby('client_id')
    ids = groupedlogs.groups.keys()
    with open(outputFilePath, 'w') as f:
        for keyid in ids:
            f.write(str(keyid) + ";")
            group = (groupedlogs.get_group(keyid).sort_values(by='timestamp')['action']).values
            for g in group:
                f.write(' '+ g)
            f.write('\n')
    f.close()

def meanshift(distmatrix, list_keys):
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(distmatrix)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(distmatrix)
    labels_ms = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels_ms)
    print(list_keys, labels_ms)
    n_clusters_ = len(labels_unique)
    print('Estimated number of clusters: %d' % n_clusters_)
    # afficher le resultats de la classification dans un fichier
    file_results_meanshift = open('temp/results_meanshift.csv', 'w')
    file_results_meanshift.write('identifiant trace' + ';' + 'class clustering' + '\n')
    i = 0
    for ligne in list_keys:
        file_results_meanshift.write(ligne + ';' + str(labels_ms[i]) + '\n')
        i += 1
    file_results_meanshift.close()
    from sklearn.metrics.cluster import homogeneity_score
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels_ms)) - (1 if -1 in labels_ms else 0)
    n_noise_ = list(labels_ms).count(-1)
    firstclasse = list(labels_ms).count(0)
    secondclasse = list(labels_ms).count(1)
    thirdclasse = list(labels_ms).count(2)
    print('Estimated number of 0 points: %d' % firstclasse)
    print('Estimated number of 1 points: %d' % secondclasse)
    print('Estimated number of 2 points: %d' % thirdclasse)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % homogeneity_score(labels))
    if n_clusters_ > 1:
        print("distance matrix", distmatrix)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(distmatrix, labels_ms))
        print("Davies bouldin score: %0.3f" % davies_bouldin_score(distmatrix, labels_ms))

    return n_clusters_, labels_ms
