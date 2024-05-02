import pandas as pd
import numpy as np
import ast
from collections import Counter
from prefixspan import PrefixSpan
from sklearn.preprocessing import MinMaxScaler
from .utils import is_exact_consecutive, getUniqueEvents, contains_pattern
# import textdistance
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
import time

start_time = time.time()


''' 
Takes a trace dataframe, returns a prefixSpan dataframe with extracted patterns as follows: 
['seqId', 'pattern', 'patternLen', 'supportCount', 'support', 'originalFlag'] 
The originalFlag == 0, if the pattern is an added 1 itemset to the patterns, 
else, it is an extracted pattern via PrefixSpan
'''

def frequent_subsequence_extraction(tracedf, actionName, outputFilePath=0, min_support_percentage=80, min_length=2):
    List_traces = list(tracedf[actionName].apply(ast.literal_eval))

    def calculateSupport(value):
        return round(value / len(List_traces), 2)

    def calculateSupportCount(pattern):
        # Initialize counters
        support_count = 0
        # Iterate through sequences
        for seq in List_traces:
            if is_exact_consecutive(pattern, seq):
                support_count += 1
        return support_count

    # calculate the min support count from the provided percentage
    min_support_count = int(min_support_percentage / 100 * len(List_traces))
    ps = PrefixSpan(List_traces).frequent(min_support_count)
    print('Before filtering, the number of extracted patterns is ', len(ps))
    # Filter out for a min_length of 2 patterns
    ps = [(support, pattern) for support, pattern in ps if len(pattern) >= min_length]
    print('After filtering, the number of extracted patterns is ', len(ps))
    # Filter for exact consecutive patterns
    exact_consecutive_patterns = [(support, pattern) for support, pattern in ps if
                                  any(is_exact_consecutive(pattern, seq) for seq in List_traces)]
    originalFlags = [1] * len(exact_consecutive_patterns)
    print('The length of exact_consecutive patterns is ', len(exact_consecutive_patterns))
    print('The final chosen patterns presented as [pattern length, (pattern)] are: ')
    print(exact_consecutive_patterns)
    # add to these patterns the 1-itemset pattern of unique events in the dataset(list of traces)
    unique_events = getUniqueEvents(List_traces)
    to_add = list(zip(list(map(calculateSupportCount, unique_events)), unique_events))
    originalFlags.extend([0] * len(to_add))
    exact_consecutive_patterns.extend(to_add)

    # add these events as patterns and support [Support Count, Pattern]
    prefixspan_result = pd.DataFrame(exact_consecutive_patterns, columns=['Support Count', 'Pattern'])
    prefixspan_result['support'] = prefixspan_result['Support Count'].apply(calculateSupport).tolist()
    prefixspan_result['patternLen'] = prefixspan_result['Pattern'].apply(len).to_list()
    prefixspan_result['seqId'] = prefixspan_result.index
    prefixspan_result.rename(columns={'Pattern': 'pattern'}, inplace=True)
    prefixspan_result.rename(columns={'Support Count': 'supportCount'}, inplace=True)
    prefixspan_result['originalFlag'] = originalFlags
    prefixspan_result = prefixspan_result[['seqId', 'pattern', 'patternLen', 'supportCount', 'support', 'originalFlag']]
    if (outputFilePath != 0):
        # prefixspan_result.to_csv(os.path.join(outputDir, 'subsequences_prefixSpan.csv'), index=False)
        prefixspan_result.to_csv(outputFilePath, index=False)
    return prefixspan_result


''' 
After finding the patterns, we will filter out the traces that have no original Patterns 
Only the filtered traces are used to : 
- compute the next footprint matrix
- compute the FSS and be used later in clustering 
To the tracedf, add a column 'hasOriginalPattern' 
- 1 is the value if the trace has any of the orginal patterns, 0 otherwise
'''

def filterTraces(tracedf, prefixspan_result, actionName):
    hasOriginalPattern = []
    patterns = prefixspan_result[prefixspan_result['originalFlag'] == 1]['pattern'].to_list()
    traces = list(tracedf[actionName].apply(ast.literal_eval))
    for trace in traces:
        Flag = False
        for pattern in patterns:
            if(contains_pattern(trace, pattern) == 1):
                Flag = True
                hasOriginalPattern.append(1)
                break
        if(Flag == False):
            hasOriginalPattern.append(0)
    tracedf['hasOriginalPattern'] = hasOriginalPattern
    return tracedf


def matrix_direct_succession(tracedf, actionName):
    tracedf = tracedf[tracedf['hasOriginalPattern'] == 1]
    List_traces = list(tracedf[actionName].apply(ast.literal_eval))

    ## compute and return an event dataframe with the form [Activity_id, Activity_Name, Activity_Freq]

    # Flatten the list of lists and count occurrences using Counter
    flat_activities = [activity for sublist in List_traces for activity in sublist]
    activity_counts = Counter(flat_activities)

    # Create a DataFrame from the Counter dictionary
    df_activity_count = pd.DataFrame(list(activity_counts.items()), columns=['activity_name', 'activity_count'])

    # Add an 'activity_id' column
    df_activity_count['activity_id'] = range(0, len(df_activity_count))

    # compute the footprint matrix
    footprint_matrix = np.zeros(shape=(len(df_activity_count), len(df_activity_count)))

    for trace in List_traces:
        for i in range(1, len(trace) - 1):
            footprint_matrix[
                df_activity_count['activity_id'][df_activity_count['activity_name'] == trace[i]].values[0]][
                df_activity_count['activity_id'][df_activity_count['activity_name'] == trace[i + 1]].values[0]] += 1

    return df_activity_count, footprint_matrix


'''
An Fss value  of an  FSS pattern is calculated as follows: 
for each 2 consecutive events in the FSS compute 
r += (freq_event1 in the all the trace file * freq_event12 in the all the trace file * succession_freq(event1 with event2))
Then for the whole FSS pattern: 
FSS = r * support_count(FSS pattern)
Then, FSS values = 1 / FSS

** only works with FSS patterns with 2+ events

'''

def compute_fss_encoding(prefixSpanRes, df_activity_count, footprint_matrix):
    pattern_FSS = []
    old_FSS = []
    for i in range(0, len(prefixSpanRes)):
        sum = 0
        pattern = prefixSpanRes.iloc[i]['pattern']
        for j in range(0, len(pattern)-1):
            r = footprint_matrix[df_activity_count['activity_id'][df_activity_count['activity_name'] == pattern[j]].values[0]][df_activity_count['activity_id'][df_activity_count['activity_name'] == pattern[j+1]].values[0]]
            sum += (r * df_activity_count['activity_count'][df_activity_count['activity_name'] == pattern[j]].values[0]  *  df_activity_count['activity_count'][df_activity_count['activity_name'] == pattern[j+1]].values[0])
        fss = int(prefixSpanRes.iloc[i]['supportCount'])
        if(len(pattern) > 1):
            fss = fss * sum
        pattern_FSS.append(fss)
        if(fss != 0):
            old_FSS.append(1 / fss)
        else:
            old_FSS.append(0)

    prefixSpanRes['FSS Encoding Values'] = MinMaxScaler().fit_transform(np.array(pattern_FSS).reshape(-1, 1)).flatten()
    prefixSpanRes['FSS Encoding Values UnScaled'] = pattern_FSS
    prefixSpanRes['FSS Encoding Values Old'] = old_FSS #baseline, Marwa's approach
    return prefixSpanRes

def replace_fss_in_trace(tracedf, actionName, prefixSpanRes, outputPath=False):
    #add a column to the tracedf which is actionName_FSSEncoded
    tracedf = tracedf[tracedf['hasOriginalPattern'] == 1]
    List_traces = list(tracedf[actionName].apply(ast.literal_eval))
    ps_sorted = prefixSpanRes.sort_values(by='patternLen', ascending=False)
    patterns = ps_sorted['pattern']
    enc = ps_sorted['FSS Encoding Values']

    #patterns in traces should be replaced by the desc order of pattern len (so start searching for longer patterns and from the rest you search
    # for smaller patterns)
    replaced_traces = []
    traces_len = []
    for trace in List_traces:
        replaced_trace = []
        i = 0
        pattern_found = False  # Flag to track if any pattern is found in the trace

        while i < len(trace):
            replaced = False
            for pattern, encoding in zip(patterns, enc):
                if trace[i:i+len(pattern)] == pattern:
                    replaced_trace.append(encoding)
                    i += len(pattern)
                    replaced = True
                    pattern_found = True  # Set the flag to True if any pattern is found
                    break
            if not replaced:
                replaced_trace.append(-1)
                i += 1
        if not pattern_found:
            replaced_trace = [-1] * len(trace)  # Replace the entire trace with -1 if no pattern is found
        replaced_traces.append(replaced_trace)
        traces_len.append(len(replaced_trace))

    tracedf[actionName+'_FSSEncoded'] = replaced_traces
    tracedf[actionName+'_FSSEncodedLen'] = traces_len
    if(outputPath):
        tracedf.to_csv(outputPath, index=False)
    return tracedf



# dictionnaire fréquences des événements dans les traces
def dict_event_frequences(counter):
    frequenceevent = {} # dictionnaire fréquence evenements
    identifiantevent = {} # dictionnaire identifiant event
    j = 0
    for i in counter:
        #print(i, ':', counter[i])
        frequenceevent[i] = counter[i]
        identifiantevent[i] = j
        j += 1
    return frequenceevent, identifiantevent


# transformation de la representation vectorielle de la trace pour l'approche encodage
# il faut reouvrir le file (trace avec sous sequence fréquente encodées)
# remplacer les gaps par la valeur 1 ( garder la taille de la trace)
def replace_gap(identifiantevent):
    f_encodage = open('./nouvelletrace-encodage.csv', "r")
    f6 = open('./nouvelletrace-ssf-gap_encodage.csv', 'w')
    tab_enco = []
    for ligne_enco in f_encodage:
        ligne_enco = ligne_enco.rstrip()
        tab_enco = ligne_enco.split(';')
        id_enco = tab_enco[0]
        f6.write(id_enco)
        f6.write(';')
        trace_enco = tab_enco[1]
        #print(type(trace_enco))
        events_idss = trace_enco.split(' ')
        #print(len(events_idss))

        for i in range(0, len(events_idss)):
            #print(events_idss[i])
            if events_idss[i] in identifiantevent.keys():
                f6.write("1")
                f6.write(' ')
            else:
                f6.write(events_idss[i])
                f6.write(' ')
        f6.write('\n')

    f_encodage.close()
    f6.close()



# Convert event logs to trace logs

def convertLogs(logFilePath, outputFilePath):
    logs = pd.read_csv(logFilePath, delimiter=';')
    groupedlogs = logs.groupby('session_id')
    ids = groupedlogs.groups.keys()
    with open(outputFilePath, 'w') as f:
        for keyid in ids:
            f.write(str(keyid) + ";")
            group = (groupedlogs.get_group(keyid).sort_values(by='@timestamp')['action']).values
            for g in group:
                f.write(' ' + g)
            f.write('\n')
    f.close()


# Extract frequent subsequence from events logs

def frequent_subsequence_extraction():
    convertLogs("Real_logs_all.csv", "Modified_events_logs.csv")
    List_traces = []
    f1 = open("Modified_events_logs.csv", "r")
    for ligne in f1:
        ligne = ligne.strip()
        tab = ligne.split(';')
        id = tab[0]
        trace = tab[1]
        List_traces.append(trace.split())
        # List_traces.append([trace])
    # print(List_traces)
    ps = PrefixSpan(List_traces)
    print(ps)
    prefixspan_result = (ps.topk(100))
    # print(prefixspan_result)
    file = open("subsequences.csv", "w")
    file.write("identifiant;Subsequence;Support;Count;nevent" + "\n")
    cmp = 1
    line = ""
    for i in prefixspan_result:
        print('listsubsequence', i)
        print(str(len(i[1])))
        # if ((len(i[1])) >= 2 ) and (len(i[1]) < 7):
        # if (len(i[1][0].split(" ")) > 1):
        line += str(cmp) + ";" + ' '.join(i[1]) + ";" + str(round(i[0] / len(List_traces), 2)) + ";" + str(
            i[0]) + ";" + str(len(i[1])) + "\n"
        # line += str(cmp) + ";" + str(i[1][0]) + ";" + str(round(i[0] / len(List_traces), 2)) + ";" + str(i[0]) + ";" + str(len(i[1][0].split(" "))) + "\n"
        file.write(line)
        line = ""
        cmp += 1
    f1.close()
    file.close()


# construire trois dict pour les ssf (id,ssf) (id,longueur) (id,fréquence)

def dict_subsequences():
    f = open("subsequences.csv", "r", encoding='utf-8')
    # construire le dictionnaire des sous séquences fréquentes
    subseq = {}
    subseq_long = {}
    subseq_count = {}
    for ligne in f:
        # ligne = ligne.rstrip()
        tab2 = ligne.split(';')
        id_ss = tab2[0]
        ss = tab2[1]
        support = tab2[2]
        count = tab2[3]
        nbevent = tab2[4]
        nbevent = nbevent[:-1]
        subseq[id_ss] = ss  # dictionnaire sous sequence et son identifiant
        subseq_long[id_ss] = nbevent  # dictionnaire identifiant ss et sa longueur
        subseq_count[id_ss] = count  # dictionnaire identifiant ss et sa fréquence d'apparition

    return subseq, subseq_long, subseq_count


# extract raw traces
def tracebrute():
    f1 = open("Modified_events_logs.csv", "r")
    ftrace = open("tracebrute.csv", "w")
    for ligne in f1:
        tab = ligne.split(';')
        id = tab[0]
        trace = tab[1]
        ftrace.write(trace)
    ftrace.close()


# counter of events in the eventlogs
def event_count(filename):
    with open(filename) as f:
        return Counter(f.read().split())


# dictionnaire fréquences des événements dans les traces
def dict_event_frequences(counter):
    frequenceevent = {}  # dictionnaire fréquence evenements
    identifiantevent = {}  # dictionnaire identifiant event
    j = 0
    for i in counter:
        # print(i, ':', counter[i])
        frequenceevent[i] = counter[i]
        identifiantevent[i] = j
        j += 1
    return frequenceevent, identifiantevent


# matrice des successions directes entre les événements
def matrix_direct_succession(frequenceevent, identifiantevent):
    footprint_matrix = np.zeros(shape=(len(frequenceevent.keys()), len(frequenceevent.keys())))
    ftrace = open("tracebrute.csv", "r")
    for t in ftrace:
        t = t.rstrip()
        events = t.split(' ')
        # print (events)
        for i in range(1, len(events) - 1):
            e1 = events[i]
            e2 = events[i + 1]
            if e1 in identifiantevent.keys():
                id1 = identifiantevent[e1]
            if e2 in identifiantevent.keys():
                id2 = identifiantevent[e2]
            footprint_matrix[id1][id2] += 1
            # footprint_matrix[
    # print(footprint_matrix)
    ftrace.close()
    return footprint_matrix


# lire le log
# parcourir les ss et remplacer les ss dans les traces par leur encodage
# encodage : données qu'on a besoin pour l'encodage
# ssf = subseq.values()
# taille de ssf = subseq_long.values()
# frequence de ssf =  subseq_count.values()
# pour chaque evenement( frequenceevent.keys()) in  ssf (subseq.values())
# filtre des sous séquences, garder que les sous séquences ayant des activités moin fréquentes
# lire la fréquence des activités dans le log total
def replace_ssf_encoding(subseq, subseq_long, subseq_count, identifiantevent, frequenceevent, footprint_matrix):
    myfile_encodage = open('nouvelletrace-encodage.csv', 'w')
    f1 = open("Modified_events_logs.csv", "r")
    for ligne in f1:
        tab = ligne.split(';')
        id = tab[0]
        trace = tab[1]
        bool = 0
        for k in subseq.keys():
            if (subseq[k] in trace) and (int(subseq_long[k]) > 1):  # tester si la ssf existe dans la trace
                fss = subseq_count[k]
                events = subseq[k].split(' ')  # parcours des evenements dans une ssf
                somme = 0
                for i in range(0, len(events) - 1):  # taille de la ssf -1
                    e1 = events[i]
                    e2 = events[i + 1]
                    id1 = identifiantevent[e1]
                    id2 = identifiantevent[e2]
                    f1 = frequenceevent[e1]
                    f2 = frequenceevent[e2]
                    r = footprint_matrix[id1][id2]
                    somme += f1 * f2 * r
                id_ss = 1 / (int(fss) * somme)

                ligne = ligne.replace(subseq[k], str(id_ss))
                bool = 1

        if (bool == 1):  # ecrire que les traces avec frequent subsequences
            myfile_encodage.write(ligne)
        bool = 0

    myfile_encodage.close()


# transformation de la representation vectorielle de la trace pour l'approche encodage
# il faut reouvrir le file (trace avec sous sequence fréquente encodées)
# remplacer les gaps par la valeur 1 ( garder la taille de la trace)
def replace_gap(identifiantevent):
    f_encodage = open('nouvelletrace-encodage.csv', "r")
    f6 = open('nouvelletrace-ssf-gap_encodage.csv', 'w')
    tab_enco = []
    for ligne_enco in f_encodage:
        ligne_enco = ligne_enco.rstrip()
        tab_enco = ligne_enco.split(';')
        id_enco = tab_enco[0]
        f6.write(id_enco)
        f6.write(';')
        trace_enco = tab_enco[1]
        # print(type(trace_enco))
        events_idss = trace_enco.split(' ')
        # print(len(events_idss))

        for i in range(0, len(events_idss)):
            # print(events_idss[i])
            if events_idss[i] in identifiantevent.keys():
                f6.write("1")
                f6.write(' ')
            else:
                f6.write(events_idss[i])
                f6.write(' ')
        f6.write('\n')

    f_encodage.close()
    f6.close()


# dictionnaire des ssf encodées avec gap =1
def new_traces_encoding():
    f6 = open('nouvelletrace-ssf-gap_encodage.csv', 'r')
    subseq_tracetransforme_encodage = {}
    for ligne_gap in f6:
        ligne_gap = ligne_gap.rstrip()
        tab_tracetransforme_encodage_gap = ligne_gap.split(';')
        id_tracessf_encodage_gap = tab_tracetransforme_encodage_gap[0]
        trace_transforme_encodage_gap = tab_tracetransforme_encodage_gap[1]
        subseq_tracetransforme_encodage[
            id_tracessf_encodage_gap] = trace_transforme_encodage_gap  # dictionnaire des traces d'encodage  avec les gap
    f6.close()
    return subseq_tracetransforme_encodage


# mettre les traces transformés sous forme d'une liste de string et ensuite en liste de vecteurs float,
# pour le calcul des distances pour approche , peut être utilisé pour les 4 approches
def new_traces_frequences_events_list(dictionnaire):
    list_values = []  # liste des valeurs
    list_keys = []  # liste des clés
    for key, value in dictionnaire.items():
        list_values.append(value)

    for key, value in dictionnaire.items():
        list_keys.append(key)
    return list_values, list_keys


# transformer chaque element de la liste des string en float, peut être utilisé pour les 4 approches
def new_traces_frequences_events_list_float(list_values):
    list_values_float = []  # liste des vecteurs de float
    for i in list_values:
        mot = i.split(' ')
        list_values_f = []
        for j in mot:
            if j != '':
                list_values_f.append(float(j))
        list_values_float.append(list_values_f)
    return list_values_float


# fonction qui retourne la taille maximale des traces transformees, peut être utilisé pour les 4 approches
def get_max_long_trace_transforme(list):
    max = 0
    for ligne in list:
        long = len(ligne)
        # print(ligne, long)
        if max < long:
            max = long
    return (max)


# ajouter des zero afin d'avoir des vecteurs de float de taille fixe, peut être utilisé pour les 4 approches
def same_length_vectors(list_values_float, max):
    cpt = 0
    for elt in list_values_float:
        cpt = cpt + 1
        print('#####################', cpt)
        for i in range(max):
            if len(elt) != max:
                # print(len(elt))
                reste = max - len(elt)
                # print(reste)
                for j in range(reste):
                    elt.append(0.0)
    return (list_values_float)


# Construct a levenshtein distance matrix for a list of strings

def get_distance_matrix_Levenshtein(list):  # 1 ere approche
    dist_matrix = np.zeros(shape=(len(list), len(list)))
    print("Starting to build levenshtein distance matrix. This will iterate from 0 until", len(list))
    normalized_levenshtein = NormalizedLevenshtein()
    for i in range(0, len(list)):
        print(i)
        for j in range(0, len(list)):
            # print(normalized_levenshtein.distance(list[i], list[j]), list[i], list[j])
            dist_matrix[i][j] = normalized_levenshtein.distance(list[i], list[j])

    return dist_matrix


# Construct a consine distance matrix

def get_distance_matrix_cosine_text_distance(list):
    print(len(list))
    dist_matrix = np.zeros(shape=(len(list), len(list)))
    print("Starting to build hamming distance matrix. This will iterate from 0 until", len(list))
    for i in range(0, len(list)):
        # print (i)
        for j in range(0, len(list)):
            dist_matrix[i][j] = textdistance.cosine.distance(list[i], list[j])
            # print(list[i], 'vs', list[j], 'dist', str(dist_matrix[i][j]))
    for i in range(0, len(list)):
        for j in range(0, len(list)):
            if i == j:
                dist_matrix[i][j] = 0
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]

    return dist_matrix


# Construct a consine distance matrix

def get_distance_matrix_jaccard_scipy(list):
    dist_matrix = np.zeros(shape=(len(list), len(list)))
    print("Starting to build jaccard distance matrix. This will iterate from 0 until", len(list))
    # cosine = Cosine(2)
    for i in range(0, len(list)):
        for j in range(0, len(list)):
            dist_matrix[i][j] = distance.jaccard(list[i], list[j])
            # if dictList_encodage_values_float[j] == dictList_encodage_values_float[i]:
            # dist_matrix[i][j] = 0.0
            # print(list[i],'vs', list[j],'dist', str(dist_matrix[i][j]))
            # print(X[i], 'vs', X[j], 'dist', str(dist_matrix[i][j]))
    return dist_matrix


# Compute clustering with MeanShift

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
    file_results_meanshift = open('results_meanshift.csv', 'w')
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
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(distmatrix, labels_ms))
        print("Davies bouldin score: %0.3f" % metrics.davies_bouldin_score(distmatrix, labels_ms))
    return n_clusters_, labels_ms


def result_clustering_dict():
    result = open('results_meanshift.csv', "r")
    subseq_result = {}  # dictionnaire des resultats de clustering
    for ligne_result in result:
        tab_result = ligne_result.split(';')
        id_case = tab_result[0]
        class_result = tab_result[1]
        subseq_result[id_case] = class_result
    result.close()
    return subseq_result


def result_clustering_real_log(subseq_result):
    reallog1 = open("all-modelise_result.csv", "w")
    reallog = open("Real_logs_all.csv", "r")
    for ligne_log in reallog:
        isLogDeleted = 0
        ligne_log = ligne_log.strip()
        tab_logreal = ligne_log.split(';')
        # ipadress = tab_logreal[1]
        date = tab_logreal[1]
        action = tab_logreal[2]
        idcase_logreal = tab_logreal[0]
        class_res = ""
        if action == "action":
            class_res = "class\n"
        else:
            if idcase_logreal in subseq_result:
                class_res = str(subseq_result[idcase_logreal])
            else:
                isLogDeleted = 1
            # print(idcase_logreal)
        if isLogDeleted == 0:
            reallog1.write(date)
            reallog1.write(';')
            reallog1.write(action)
            reallog1.write(';')
            reallog1.write(idcase_logreal)
            reallog1.write(';')
            # reallog1.write(ipadress)
            # reallog1.write(";")
            reallog1.write(class_res)
    reallog1.close()
    reallog.close()


# second step : generate files of logs with the number of the found clusters

def reallog_result_clustering():
    models_file = open("all-modelise_result.csv", "r")
    content = models_file.readlines()
    content = content[1:]
    f = open("class_0.csv", 'w')
    f1 = open("class_1.csv", 'w')
    f2 = open("class_2.csv", 'w')
    f3 = open("class_3.csv", 'w')
    f.write("date;action;idcase_logreal;class_res" + "\n")
    f1.write("date;action;idcase_logreal;class_res" + "\n")
    f2.write("date;action;idcase_logreal;class_res" + "\n")
    f3.write("date;action;idcase_logreal;class_res" + "\n")
    # f3.write("date;action;idcase_logreal;;ipadress;class_res" + "\n")
    for ligne in content:
        ligne = ligne.strip()
        # print(ligne)
        tab_models = ligne.split(';')
        date_models = tab_models[1]
        # print(date_models)
        action_models = tab_models[2]
        idcase_models = tab_models[0]
        # ipadress_models = tab_models[3]
        class_models = tab_models[3]
        # print(class_models)
        if class_models == "0":
            f.write(ligne)
            f.write('\n')
        elif class_models == "1":
            f1.write(ligne)
            f1.write('\n')
        elif class_models == "2":
            f2.write(ligne)
            f2.write('\n')
        else:
            f3.write(ligne)
            f3.write('\n')

    # for i in range(n_clusters_):
    # fichier = "log_modelise/fichier{}.csv".format(i+1)

    f.close()
    f1.close()
    f2.close()
    f3.close()
    models_file.close()

