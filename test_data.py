import numpy as np
from numpy.random import default_rng

#generate dataset 
def TestLinear(w, b, n_A, n_B, margin, **kwargs):
    """Generates linearly separable test data with a specified margin."""
    seed = kwargs.get("seed", 18)
    shape = kwargs.get("shape", 1.0)
    scale = kwargs.get("scale", 1.0)
    sigma = kwargs.get("sigma", 1.0)
    
    d = w.size
    norm_w = np.linalg.norm(w)
    w = w / norm_w
    b = b / norm_w
    
    rng = default_rng(seed)
    list_A, list_B = [], []
    
    for _ in range(n_A):
        vec = rng.normal(size=d, scale=sigma)
        dist = rng.gamma(shape, scale)
        vec += -np.inner(vec, w) * w
        vec += (dist + margin - b) * w
        list_A.append(vec)
    
    for _ in range(n_B):
        vec = rng.normal(size=d, scale=sigma)
        dist = rng.gamma(shape, scale)
        vec += -np.inner(vec, w) * w
        vec += (-b - dist - margin) * w
        list_B.append(vec)
    
    vec = rng.normal(size=d, scale=sigma)
    vec += -np.inner(vec, w) * w
    supp_A = rng.integers(0, n_A)
    list_A[supp_A] = vec + (margin - b) * w
    supp_B = rng.integers(0, n_B)
    list_B[supp_B] = vec + (-b - margin) * w
    
    return np.array(list_A), np.array(list_B)