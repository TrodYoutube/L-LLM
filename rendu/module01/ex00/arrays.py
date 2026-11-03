import numpy as np

def zeros(r, c):
    return (np.zeros((r, c)))
    

def identity(n):
    return (np.eye(n))

def arange_matrix(r, c):
    return (np.arange(r*c).reshape(r, c))

def describe(a):
    return (a.shape, a.ndim, a.size)