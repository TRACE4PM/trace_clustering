import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from .FSS_encoding.utils import same_length_vectors
from .FSS_encoding.data_preparation import (frequent_subsequence_extraction,
                                            filterTraces, matrix_direct_succession,
                                            compute_fss_encoding, replace_fss_in_trace)


def getBinaryRep(tracedf, caseIDcol, actionName):
    List_traces = list(tracedf[actionName])
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(List_traces).transform(List_traces)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df = df.astype(int)
    df[caseIDcol] = tracedf[caseIDcol].to_list()

    # dropping the client_id col to get only the binary vectors
    binary_vectors = df.drop(columns=['client_id']).values
    return binary_vectors


def getFreqRep(tracedf, caseIDcol, actionName):
    List_traces = list(tracedf[actionName])
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    _ = te.fit(List_traces).transform(List_traces)
    activitydf = pd.DataFrame()
    for activity in te.columns_:
        if activity != 'index':  # Exclude 'index' column if present in encdf
            count_column_name = activity
            activitydf[count_column_name] = (tracedf[actionName]).apply(
                lambda x: x.count(activity) if activity in x else 0).to_list()
    # Replace True with 1 and False with 0
    activitydf = activitydf.astype(int)
    activitydf[caseIDcol] = tracedf[caseIDcol].to_list()

    # dropping the client_id col to get only the values of the vectors
    freq_vectors = activitydf.drop(columns=['client_id']).values
    return freq_vectors


def extractRelativeFreq(tracedf, caseIDcol, actionName):
    List_traces = list(tracedf[actionName])
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    _ = te.fit(List_traces).transform(List_traces)
    activitydf = pd.DataFrame()
    for activity in te.columns_:
        if activity != 'index':
            count_column_name = activity
            activitydf[count_column_name] = (tracedf[actionName]).apply(
                lambda x: x.count(activity) / len(x) if activity in x else 0).to_list()
    # Replace True with 1 and False with 0
    activitydf[caseIDcol] = tracedf[caseIDcol].to_list()

    # dropping the client_id col to get only the values of the vectors
    relfreq_vectors = activitydf.drop(columns=['client_id']).values
    return relfreq_vectors


def get_FSS_encoding(tracedf, trace_col, min_support_percentage, min_length):
    """
        FSS encoding improved version
        Returns a list of FSS encoded vectors
    """
    prefixSpanRes = frequent_subsequence_extraction(tracedf, trace_col, min_support_percentage=min_support_percentage,
                                                    min_length=min_length)
    filteredTraces = filterTraces(tracedf, prefixSpanRes, trace_col)
    filteredTraces = filteredTraces[filteredTraces['hasOriginalPattern'] == 1]
    print('New Length of tracedf ', len(filteredTraces))
    # now apply FSS on filtered traces using the computed prefixSpan result of patterns
    df_activity_count, footprint_matrix = matrix_direct_succession(filteredTraces, 'SemanticTrace')
    prefixSpanRes = compute_fss_encoding(prefixSpanRes, df_activity_count, footprint_matrix)
    replaced_trace = replace_fss_in_trace(filteredTraces, 'SemanticTrace', prefixSpanRes)

    # fss_encoded_vectors = list(replaced_trace['trace_FSSEncoded'])
    return replaced_trace


# ************ chosing the vector representation *********

def vectorRepresentation(vector_representation, traces):
    vectors = []
    if vector_representation == "binary representation":
        vectors = getBinaryRep(traces, "client_id", "trace")
    elif vector_representation == "frequency representation":
        vectors = getFreqRep(traces, "client_id", "trace")
    elif vector_representation == "relative frequency representation":
        vectors = extractRelativeFreq(traces, "client_id", "trace")
    # elif vector_representation == "fss encoding":
    #     vectors = get_FSS_encoding(traces, "client_id", "trace" )
    return vectors
