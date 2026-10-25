import numpy as np

from _utils import check, close, load, need, no_loops


def ex00(d):
    m = load(d, "arrays.py")
    need(m, "zeros", "identity", "arange_matrix", "describe")
    z = m.zeros(2, 3)
    check(isinstance(z, np.ndarray), "zeros doit renvoyer un np.ndarray")
    check(z.shape == (2, 3) and np.all(z == 0), "zeros(2, 3) incorrect")
    check(np.array_equal(m.identity(3), np.eye(3)), "identity(3) incorrect")
    a = m.arange_matrix(3, 4)
    check(a.shape == (3, 4), "arange_matrix(3, 4) doit avoir la forme (3, 4)")
    check(np.array_equal(a, np.arange(12).reshape(3, 4)), "arange_matrix(3, 4) doit contenir 0..11 ligne par ligne")
    check(m.describe(np.zeros((2, 5, 3))) == ((2, 5, 3), 3, 30), "describe doit renvoyer (shape, ndim, size)")


def ex01(d):
    no_loops(d, "vectorize.py")
    m = load(d, "vectorize.py")
    need(m, "dot", "mat_mul", "row_norms", "normalize_rows")
    u, v = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
    check(close(m.dot(u, v), 32), "dot incorrect")
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    check(close(m.mat_mul(A, B), A @ B), "mat_mul incorrect")
    X = np.array([[3.0, 4], [0, 2], [1, 0]])
    check(close(m.row_norms(X), [5, 2, 1]), "row_norms incorrect")
    n = m.normalize_rows(X)
    check(n.shape == X.shape, "normalize_rows doit garder la forme de X")
    check(close(np.linalg.norm(n, axis=1), 1), "normalize_rows : chaque ligne doit avoir une norme de 1")
    check(close(n[0], [0.6, 0.8]), "normalize_rows incorrect")


def ex02(d):
    no_loops(d, "broadcasting.py")
    m = load(d, "broadcasting.py")
    need(m, "add_bias", "standardize", "pairwise_sq_dist")
    X = np.array([[2.0, 3], [4, 5], [6, 7]])
    Xb = m.add_bias(X)
    check(Xb.shape == (3, 3), "add_bias doit ajouter UNE colonne")
    check(close(Xb[:, 0], 1) and close(Xb[:, 1:], X), "add_bias : la colonne de 1 doit être la PREMIÈRE")
    rng = np.random.default_rng(0)
    R = rng.normal(5, 3, size=(50, 4))
    S = m.standardize(R)
    check(close(S.mean(axis=0), 0) and close(S.std(axis=0), 1), "standardize : chaque COLONNE doit avoir moyenne 0, std 1")
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    expected = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    D = m.pairwise_sq_dist(A, B)
    check(D.shape == (5, 7), f"pairwise_sq_dist : forme attendue (5, 7), reçu {np.shape(D)}")
    check(close(D, expected), "pairwise_sq_dist incorrect")


def ex03(d):
    no_loops(d, "indexing.py")
    m = load(d, "indexing.py")
    need(m, "positives", "one_hot", "argmax_rows", "accuracy", "shuffle_together")
    check(np.array_equal(m.positives(np.array([-1, 3, 0, 5, -2])), [3, 5]), "positives incorrect")
    oh = m.one_hot(np.array([0, 2, 1, 2]), 3)
    check(np.array_equal(oh, [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]), "one_hot incorrect")
    check(np.array_equal(m.argmax_rows(np.array([[1, 9, 3], [7, 2, 1]])), [1, 0]), "argmax_rows incorrect")
    check(close(m.accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 4])), 0.75), "accuracy incorrect")
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10) * 10
    Xs, ys = m.shuffle_together(X, y, seed=42)
    perm = np.random.default_rng(42).permutation(10)
    check(np.array_equal(Xs, X[perm]) and np.array_equal(ys, y[perm]), "shuffle_together : utilise np.random.default_rng(seed).permutation")
    check(np.array_equal(X, np.arange(20).reshape(10, 2)), "shuffle_together ne doit pas modifier X")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03)]
