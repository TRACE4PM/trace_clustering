'''
returns a df with the structure [seqId,pattern,	patternLen,	supportCount,support]
'''


def is_exact_consecutive(pattern, sequence):
    # Check if the pattern occurs exactly in the same order and consecutively in the sequence
    pattern_len = len(pattern)
    sequence_len = len(sequence)

    for i in range(sequence_len - pattern_len + 1):
        if sequence[i: i + pattern_len] == pattern:
            return True
    return False


''' 
the following function takes a list of traces that looks like this [[t1], [t2], ... [tn]] and extracts the unique events as follows
the output looks like this: 
[['Apply'], ['Assess_A'], ['Assess_P'], ['Interact'], ['Exercise_A'], ['Expand'], ['Exercise_P'], ['View'], ['Feedback'], ['Study_P'], ['Study_A'], ['Revise']]
'''


def getUniqueEvents(List_traces):
    # Flatten the list of lists
    flat_list = [word for sublist in List_traces for word in sublist]
    # Get unique events (unique words)
    unique_events = set(flat_list)
    # Create a list of lists with each unique event in a separate list
    lists_of_unique_events = [[word] for word in unique_events]

    return lists_of_unique_events

def contains_pattern(sequence, pattern):
    j = 0
    for i in range(len(sequence)):
        if sequence[i] == pattern[j]:
            j += 1
            if j == len(pattern):
                return 1

    return 0
