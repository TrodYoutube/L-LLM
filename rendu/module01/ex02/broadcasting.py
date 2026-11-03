import numpy as np

def add_bias(X):
    tab = np.ones((X.shape[0], 1))
    X = np.concatenate([tab, X], axis=1)
    return (X)

def standardize(X):
    return ((X - np.mean(X, axis=0)) / np.std(X, axis=0))

# A = (2, 2) B = (3, 2)
# A[:, None, :] devient : A = (2, 1, 3)

def pairwise_sq_dist(A, B):
    return (np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2))




if __name__ == "__main__":
    A = np.array([[2, 20], [3, 30]])
    B = np.array([[8, 7], [9, 5], [5, 4]])
    print(np.shape(A[:, None, :]))
    print(np.shape(B[None, :, :]))
    print()
    print(pairwise_sq_dist(A, B))