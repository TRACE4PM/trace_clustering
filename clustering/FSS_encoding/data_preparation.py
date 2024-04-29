import pandas as pd
import numpy as np
import ast
from collections import Counter
from prefixspan import PrefixSpan
from sklearn.preprocessing import MinMaxScaler
from .utils import is_exact_consecutive, getUniqueEvents, contains_pattern

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