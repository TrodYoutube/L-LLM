import numpy as np

def dot(u, v):
    return (np.sum(u * v))

def mat_mul(A, B):
    return (A @ B)

def row_norms(X):
    return (np.sqrt(np.sum(X ** 2, axis=1)))

def normalize_rows(X):
    return ((X[:]) / row_norms(X).reshape(-1, 1))

if __name__ == "__main__":
    X = np.array([[3.0, 4], [0, 2], [1, 0]])
    n = normalize_rows(X)
    print(n.shape)
    print(n[0])