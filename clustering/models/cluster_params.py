from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import Optional, Union


class ClusteringMethod(str, Enum):
    Agglomerative = "Agglomerative"
    DBSCAN = "DBSCAN"


class DistanceMeasure(str, Enum):
    Jaccard = "Jaccard"
    Hamming = "Hamming"
    Cosine = "Cosine"


class VectorRepresentation(str, Enum):
    Binary = "Binary Representation"
    Frequency = "Frequency Representation"
    RelativeFrequency = "Relative Frequency Representation"


class ClusteringParams(BaseModel):
    """
        epsilon and min_samples parameters for dbscan algorithm
        nbr_cluster and linkage critetia for agglomerative algorithm
    """
    epsilon: Optional[float] = None
    min_samples: Optional[int] = None
    nbr_clusters: Optional[int] = None
    linkage: Optional[str] = None

