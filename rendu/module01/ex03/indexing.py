import numpy as np

def positives(a):
    return (a[a > 1])

def one_hot(labels, n_classes):
    return(np.eye(n_classes)[labels])

def argmax_rows(X):
    return (np.argmax(X, axis=1))

def accuracy(y_pred, y_true):
    return ((y_pred == y_true).mean())

def shuffle_together(X, y, seed):
    perm = np.random.default_rng(seed).permutation(len(X))
    return (X[perm], y[perm])



if __name__ == "__main__":
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    perm = np.random.default_rng(42).permutation(len(X))
    print(X)
    print(perm)
    print(X[perm])