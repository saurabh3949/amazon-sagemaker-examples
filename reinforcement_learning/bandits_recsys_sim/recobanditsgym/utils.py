import numpy as np
from sklearn.preprocessing import normalize


def uniform_feature_vec(dim: int):
    vec = np.random.rand(1, dim)
    vec = normalize(vec, norm='l2', axis=1)
    return vec