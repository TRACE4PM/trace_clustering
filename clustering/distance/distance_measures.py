import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import jaccard
from similarity.normalized_levenshtein import NormalizedLevenshtein


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
                string1 = traces.iloc[i]['trace']
                string2 = traces.iloc[j]['trace']
                lev_dist = normalized_levenshtein.distance(string1, string2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(lev_dist, 3)
                j = j + 1
            distance_matrix[i][j] = 0
            j = 0
    return distance_matrix


################# Distance measures for feature based clustering ########################

# these distance measures only use the vector representation of the features to calculate the distance matrix

def cosine_distance(vectors):
    max_size = len(vectors)
    distance_matrix = np.empty((max_size, max_size), float)
    for i in range(max_size):
        for j in range(i, max_size):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                v1 = vectors[i]
                v2 = vectors[j]
                dist = distance.cosine(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)

    return distance_matrix


def jaccard_distance(vectors):
    max_size = len(vectors)
    distance_matrix = np.empty((max_size, max_size), float)
    for i in range(max_size):
        for j in range(i, max_size):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                v1 = vectors[i]
                v2 = vectors[j]
                dist = jaccard(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)
    return distance_matrix


def hamming_distance(vectors):
    max_size = len(vectors)
    distance_matrix = np.empty((max_size, max_size), float)

    for i in range(max_size):
        for j in range(i, max_size):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                v1 = vectors[i]
                v2 = vectors[j]
                dist = distance.hamming(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)

    return distance_matrix
