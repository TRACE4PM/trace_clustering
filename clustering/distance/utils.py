import os
import pandas as pd


def number_traces(path):
    """
    calculates the number of traces in the log files of each cluster
    """
    files_to_process = []
    # opens the files named cluster_log where the logs are stored
    for filename in os.listdir(path):
        if filename.startswith("cluster_log_"):
            file_path = os.path.join(path, filename)
            with open(file_path, 'rb') as f:
                files_to_process.append(file_path)
    # iterating over each file and grouping the actions of each client
    nb_clusters = len(files_to_process)
    for i in range(nb_clusters):
        df = pd.read_csv(files_to_process[i], sep=";")
        traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
        print(files_to_process[i], len(traces))
