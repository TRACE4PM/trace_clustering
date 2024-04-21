from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import Optional, Union

class ClusteringMethod(str, Enum):
    Agglomerative = "Agglomerative"
    DBSCAN = "DBScan"

class ClusteringParams(BaseModel):
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    n_clusters: Optional[int] = None
    linkage: Optional[str] = None

