import numpy as np
import pandas as pd
from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics import davies_bouldin_score, silhouette_score


def same_length_vectors(list_values_float):
    max = max_length(list_values_float)
    print('max length vector', max)
    cpt = 0
    for elt in list_values_float:
        cpt = cpt + 1
        for i in range(max):
            if len(elt) != max:
                reste = max - len(elt)
                for j in range(reste):
                    elt.append(0)
    return (list_values_float)


def max_length(list_values_float):
    max = 0
    for i in range(0, len(list_values_float)):
        if len(list_values_float.iloc[i]) > max:
            max = len(list_values_float.iloc[i])
    return max


def convertLogs(logFilePath, outputFilePath):
    logs = pd.read_csv(logFilePath, delimiter=';')
    groupedlogs = logs.groupby('client_id')
    ids = groupedlogs.groups.keys()
    with open(outputFilePath, 'w') as f:
        for keyid in ids:
            f.write(str(keyid) + ";")
            group = (groupedlogs.get_group(keyid).sort_values(by='timestamp')['action']).values
            for g in group:
                f.write(' ' + g)
            f.write('\n')
    f.close()

