# import Levenshtein
from scipy.spatial.distance import jaccard
from .utils import Normal_lev
import numpy as np
from Levenshtein import distance as lev
from scipy.spatial import distance


def cosine_distance(vectors):
    max_size = len(vectors)
    distance_matrix = np.empty((max_size, max_size), float)

    for i in range(max_size):
        for j in range(i, max_size):
            if (i == j):
                distance_matrix[i][j] = 0
            else:
                v1 = vectors[i]
                v2 = vectors[j]
                dist = distance.cosine(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)

    return distance_matrix


def jaccard_distance(binary_vectors):
    max_size = len(binary_vectors)
    distance_matrix = np.empty((max_size, max_size), float)

    for i in range(max_size):
        for j in range(i, max_size):
            if (i == j):
                distance_matrix[i][j] = 0
            else:
                v1 = binary_vectors[i]
                v2 = binary_vectors[j]
                dist = jaccard(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)

    return distance_matrix


def levenshtein(traces):
    matrix_size = len(traces)
    distance_matrix = np.empty((matrix_size, matrix_size), float)
    j = 0
    for i in range(0, matrix_size):
        if (i == j):
            distance_matrix[i][j] = 0
        else:
            while (j < i):
                string1 = traces.iloc[i]['trace']
                string2 = traces.iloc[j]['trace']
                lev_dist = lev(string1, string2)
                dist_norm = Normal_lev(lev_dist, string1, string2)
                distance_matrix[i][j] = distance_matrix[j][i] = dist_norm
                j = j + 1
            distance_matrix[i][j] = 0
            j = 0
    return distance_matrix


def hamming_distance(vectors):
    max_size = len(vectors)
    distance_matrix = np.empty((max_size, max_size), float)

    for i in range(max_size):
        for j in range(i, max_size):
            if (i == j):
                distance_matrix[i][j] = 0
            else:
                v1 = vectors[i]
                v2 = vectors[j]
                dist = distance.hamming(v1, v2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist, 3)

    return distance_matrix
