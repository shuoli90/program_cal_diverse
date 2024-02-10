import numpy as np
from typing import List

def cluster(outputs, **kwargs):
    results = np.unique(outputs, axis=0, return_counts=True)
    return results