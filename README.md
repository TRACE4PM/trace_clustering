# TRACE4PM - Trace clustering

## Installation and Configuration

This project is part of the [TRACE4PM stack](https://github.com/TRACE4PM) and is intended to be used with the TRACE4PM API.

# Trace Clustering microservice 

This microservice implements two approaches for trace clustering: 
`Trace-Based Clustering` and `Vector-Based Clustering`. The service is designed to cluster sequences of actions (traces) performed by the users, extracted from log files. 
Below are the details for each approach:

## Trace-Based Clustering 
This approach clusters traces based on their similarity using a distance matrix computed with the Levenshtein distance metric.

`def trace_based_clustering(file_path, clustering_method, params):`
### Parameters
- **file_path**: Path to the input CSV file containing the traces data.
- **clustering_method**: The clustering method to be used. Supported methods are:
  - "DBSCAN": Density-Based Spatial Clustering of Applications with Noise.
  - "Agglomerative": Agglomerative hierarchical clustering.
- **params**: Additional parameters needed for clustering.
  - `eps` and `min_samples` : epsilon and min_samples for DBSCAN
  - 'n_clusters' and 'linkage' : number of clusters and Linkage criteria for Agglomerative clustering.
### Returns
A dictionary containing clustering evaluation metrics:
- **Davies bouldin**: Davies-Bouldin score.
- **Silhouette**: Silhouette score.
- **Silhouette of each cluster**: Silhouette scores for each generated cluster.

## Vector-Based Clustering
This approach first represents traces as vectors using various encoding techniques such as `Binary Representation`, `Frequency based Representation`, or `Relative Frequency`. 
Then, it computes a distance matrix based on the chosen distance measure (e.g., cosine, Jaccard, or Hamming distance). 
Finally, it applies clustering using DBSCAN or Agglomerative clustering.

`def vector_based_clustering(file_path, vector_representation, clustering_method, params) `
### Parameters
- **file_path**: Path to the input CSV file containing the traces data.
- **vector_representation**: use different techniques to represent the traces as vectors:
  - "binary based": Binary representation.
  - "frequency based": Frequency-based representation.
  - "relative frequency": Relative frequency representation.
- **clustering_method**: The clustering method to be used. Supported methods are:
  - "DBSCAN": Density-Based Spatial Clustering of Applications with Noise.
  - "Agglomerative": Agglomerative hierarchical clustering.
- **params**: Additional parameters needed for clustering and vector representation. For example, distance metric for vectors and clustering parameters.
### Returns 
A dictionary containing clustering evaluation metrics similar to the Trace-Based Clustering approach.



### Auteur(s)

  - Amira Ania DAHACHE

