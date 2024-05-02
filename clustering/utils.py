from sklearn.metrics import silhouette_samples
import numpy as np
import csv
import pandas as pd


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
        cluster_silhouette= silhouette_vals[cluster_mask].mean()
        cluster_silhouette_scores.append(cluster_silhouette)

    return cluster_silhouette_scores


def save_clusters(log_df,clusters, traces):
    """
        Prepares the data of each cluster 'cluster_id' and
        'cluster_info_df' which is a dataframe that stores the traces of each client_id in the cluster
         to use them in  clusters_to_logs function
        """

    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        cluster_traces = traces[clusters == cluster_id]['trace']
        client_id = traces[clusters == cluster_id]['client_id']
        cluster_info_df = pd.DataFrame({'client_id': client_id, 'traces': cluster_traces})
        clusters_to_logs(log_df,cluster_id, cluster_info_df)

def clusters_to_logs(original_logs_df, cluster_id, cluster_info_df):
    """
        Iterates over the traces of each client in the cluster and filters them depending on the original
        log file, and saving them as log files to each cluster in a CSV format

         """

    file_path = f'temp/logs/cluster_log_{cluster_id}.csv'

    # Iterate over cluster_info_df and create log files for each cluster
    with open(file_path, 'w') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['client_id', 'action', 'timestamp'])
        for index, row in cluster_info_df.iterrows():
            client_ids = row['client_id']
            traces = row['traces']
            client_ids_list = client_ids.split(',')

            # Filter original logs for client_ids in the current cluster
            cluster_logs_df = original_logs_df[original_logs_df['client_id'].isin(client_ids_list)]

            # Filter logs based on traces in the cluster
            filtered_logs_df = pd.DataFrame(columns=['client_id', 'action', 'timestamp'])
            filtered_logs_df['timestamp'] = pd.to_datetime(filtered_logs_df['timestamp'], format ='%d/%m/%y, %H:%M', utc=True)
            filtered_logs_df = pd.concat([filtered_logs_df, cluster_logs_df[cluster_logs_df['action'].isin(traces)]],
                                         ignore_index=True)

            # Write the cluster logs to a CSV file
            filtered_logs_df.to_csv(file, sep=';', index=False,header=False, mode='a')


