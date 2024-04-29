from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def getBinaryRep(tracedf, caseIDcol, actionName):
    List_traces = list(tracedf[actionName])
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(List_traces).transform(List_traces)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df = df.astype(int)
    df[caseIDcol] = tracedf[caseIDcol].to_list()

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
            activitydf[count_column_name] = (tracedf[actionName]).apply(lambda x: x.count(activity) if activity in x else 0).to_list()
    # Replace True with 1 and False with 0
    activitydf = activitydf.astype(int)
    activitydf[caseIDcol] = tracedf[caseIDcol].to_list()
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
            activitydf[count_column_name] = (tracedf[actionName]).apply(lambda x: x.count(activity)/ len(x) if activity in x else 0).to_list()
    # Replace True with 1 and False with 0
    activitydf[caseIDcol] = tracedf[caseIDcol].to_list()
    relfreq_vectors = activitydf.drop(columns=['client_id']).values

    return relfreq_vectors


