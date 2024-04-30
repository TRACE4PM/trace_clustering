# clustering


## Add your files

```
cd existing_repo
git remote add origin https://gitlab.univ-lr.fr/trace_clustering/services/clustering.git
git branch -M main
git push -uf origin main
```

## Installation and Configuration

- Pour utiliser le service clustering dans le projet, il faut juste l'installer comme un module python. Dans notre cas, avec poetry, il suffit d'exécuter les différentes étapes suivantes :
    - Inclure le lien du dépôt du service dans le fichier de configuration (pyproject.toml) du projet dans la section [tool.poetry.dependencies] de la manière suivante (discover = {git = "https://gitlab.univ-lr.fr/trace_clustering/services/clustering.git"})
    - Installer le service comme un module avec la commande suivante (poetry add discover)
    - Importer les fonctions du service comme on importe n'importe quel module python par exemple (from clustering.main import trace_based_clustering) et importer l'approche de clustering que vous voulez utiliser

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
- **params**: Additional parameters needed for clustering. For example, epsilon and min_samples for DBSCAN or the number of clusters and Linkage criteria for Agglomerative clustering.
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




# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

### Contributions et Améliorations

Les contributions à ce projet sont les bienvenues. Si vous souhaitez apporter des améliorations, veuillez suivre les étapes suivantes :

    Forkez le dépôt et clonez votre propre copie.
    Créez une branche pour vos modifications : git checkout -b feature/ma-nouvelle-fonctionnalite
    Effectuez les modifications nécessaires et testez-les de manière approfondie.
    Soumettez une pull request en expliquant en détail les modifications apportées et leur impact.

### Auteur(s)

    - Amira Ania DAHACHE

### Licence

Ce projet est sous licence L3I.
