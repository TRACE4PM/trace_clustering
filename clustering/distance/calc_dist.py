# import Levenshtein
from scipy.spatial.distance import jaccard
from .utils import Normal_lev
import numpy as np
from Levenshtein import distance as lev

# calcul distance between vectors => vector based clust

def jaccard_distance(feature1, feature2):
    keys = sorted(set(feature1.keys()) | set(feature2.keys()))
    vector1 = np.array([feature1.get(key, 0) for key in keys], dtype=bool)
    vector2 = np.array([feature2.get(key, 0) for key in keys], dtype=bool)
    distance = jaccard(vector1, vector2)

    return round(distance,3)

def levenshtein(traces):
    matrix_size = len(traces)
    distance_matrix = np.empty((matrix_size, matrix_size), float)
    j = 0
    for i in range(0, matrix_size):
        if(i == j):
            distance_matrix[i][j] = 0
        else:
            while(j < i):
                string1 = traces.iloc[i]['trace']
                string2 = traces.iloc[j]['trace']
                lev_dist = lev(string1,string2)
                dist_norm = Normal_lev(lev_dist, string1, string2)
                distance_matrix[i][j] = distance_matrix[j][i] = dist_norm
                j = j + 1
            distance_matrix[i][j] = 0
            j = 0
    return distance_matrix