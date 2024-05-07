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
                if type(traces) is list:
                    # for fss encoding the vectors are a list
                    lev_dist = normalized_levenshtein.distance(traces[i], traces[j])
                else:
                    lev_dist = normalized_levenshtein.distance(traces.iloc[i]['trace'], traces.iloc[j]['trace'])
                distance_matrix[i][j] = distance_matrix[j][i] = lev_dist
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
                distance_matrix[i][j] = distance_matrix[j][i] = dist

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
                distance_matrix[i][j] = distance_matrix[j][i] = dist
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
                distance_matrix[i][j] = distance_matrix[j][i] = dist

    return distance_matrix


def distanceMeasures(vectors, distance):
    distance_matrix = []
    if distance == "hamming":
        distance_matrix = hamming_distance(vectors)
    elif distance == "jaccard":
        distance_matrix = jaccard_distance(vectors)
    elif distance == "cosine":
        distance_matrix = cosine_distance(vectors)

    return distance_matrix
