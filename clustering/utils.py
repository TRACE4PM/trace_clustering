import csv
import numpy as np
import os
import pandas as pd
from sklearn.metrics import silhouette_samples


def silhouette_clusters(distance_matrix, cluster_assignement):
    """
    Returns: a list of silhouette scores calculated for each cluster
    """
    silhouette_vals = silhouette_samples(distance_matrix, cluster_assignement)
    cluster_silhouette_scores = []
    unique_labels = set(cluster_assignement)
    # iterates over unique_labels of each cluster
    for label in unique_labels:
        cluster_mask = (cluster_assignement == label)
        cluster_silhouette = silhouette_vals[cluster_mask].mean()
        cluster_silhouette_scores.append(cluster_silhouette)

    return cluster_silhouette_scores


def save_clusters(log_df, clusters, traces):
    """
        Prepares the data of each cluster 'cluster_id' and
        'cluster_info_df' which is a dataframe that stores the traces of each client_id in the cluster
         to use them in  clusters_to_logs function
        """
    # remove any existing file in the directory
    empty_directory("temp/logs/")

    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        cluster_traces = traces[clusters == cluster_id]['trace']
        client_id = traces[clusters == cluster_id]['client_id']
        cluster_info_df = pd.DataFrame({'client_id': client_id, 'traces': cluster_traces})
        clusters_to_logs(log_df, cluster_id, cluster_info_df)


def clusters_to_logs(original_logs_df, cluster_id, cluster_info_df):
    """
    Iterates over the traces of each client in the cluster and filters them depending on the original
    log file, and saving them as log files to each cluster in a CSV format.
    """
    file_path = f'temp/logs/cluster_log_{cluster_id}.csv'

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['client_id', 'action', 'timestamp', 'cluster_id'])

        for index, row in cluster_info_df.iterrows():
            client_ids = row['client_id']
            traces = row['traces']
            client_ids_list = client_ids.split(',')

            # Filter original logs for client_ids in the current cluster
            cluster_logs_df = original_logs_df[original_logs_df['client_id'].isin(client_ids_list)]

            # Filter logs based on traces in the cluster
            filtered_logs_df = cluster_logs_df[cluster_logs_df['action'].isin(traces)].copy()

            # Add cluster_id column
            filtered_logs_df['cluster_id'] = cluster_id

            # Write the filtered logs to the CSV file
            filtered_logs_df.to_csv(file, sep=';', index=False, header=False, mode='a')


def number_traces(path):
    """
    calculates the number of traces in the log files of each cluster
    """
    nb_traces = {"Cluster n": [], "Number of traces": [],
                 "Number unique traces": []}  # Initialize dictionary to store cluster info
    for filename in os.listdir(path):
        if filename.startswith("cluster_log_"):
            cluster_num = int(filename.split("_")[2].split(".")[0])  # Extract cluster number from filename
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, sep=";")
            # Group by client_id and aggregate actions into lists
            traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
            data = np.array(traces['trace'])
            unique_traces = np.unique(data)
            nb_traces["Cluster n"].append(cluster_num)  # Append cluster number
            nb_traces["Number of traces"].append(len(data))  # Append number of traces
            nb_traces["Number unique traces"].append(len(unique_traces))

    # Sort based on cluster ID
    sorted_indices = np.argsort(nb_traces["Cluster n"])
    sorted_nb_traces = {key: [nb_traces[key][i] for i in sorted_indices] for key in nb_traces}

    return sorted_nb_traces

def empty_directory(directory_path):
    # Remove the files already existing a directory
    if os.path.exists(directory_path):
        files = os.listdir(directory_path)
        for file in files:
            os.remove(os.path.join(directory_path, file))


def save_clusters_fss(nbr_clusters,df, result_df):
    empty_directory('temp/logs')
    # Save traces of each cluster into separate CSV files
    for cluster_id in range(nbr_clusters):
        cluster_indices = result_df[result_df['cluster_id'] == cluster_id].index
        cluster_traces = df.iloc[cluster_indices][['client_id', 'action', 'timestamp']]
        cluster_traces['timestamp'] = pd.to_datetime(cluster_traces['timestamp'], format='%d/%m/%y, %H:%M',utc=True)
        cluster_traces['cluster_id'] = cluster_id
        cluster_traces.to_csv(f'temp/logs/cluster_log_{cluster_id}.csv', sep=';', index=False)
