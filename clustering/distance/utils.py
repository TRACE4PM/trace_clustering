import os
import pandas as pd

def Normal_lev(distance,s1,s2):
    dist = distance / max(len(s1),len(s2))
    return round(dist, 3)



def number_traces(path):
  files_to_process = []
  for filename in os.listdir(path):
      if filename.startswith("cluster_log_"):
          file_path = os.path.join(path, filename)
          with open(file_path, 'rb') as f:
              files_to_process.append(file_path)

  nb_clusters = len(files_to_process)
  for i in range(nb_clusters):
      df = pd.read_csv(files_to_process[i], sep=";")
      traces = df.groupby("client_id")["action"].apply(list).reset_index(name='trace')
      print(files_to_process[i], len(traces))
