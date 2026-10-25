import numpy as np

from _utils import check, close, grad_check, load, need, no_loops


def _data(seed=0, m=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, 2))
    y = (X[:, 0] + 2 * X[:, 1] - 0.5 > 0).astype(float)
    return X, y


def ex00(d):
    no_loops(d, "sigmoid.py")
    m = load(d, "sigmoid.py")
    need(m, "sigmoid")
    z = np.array([-2.0, 0, 2])
    check(close(m.sigmoid(z), 1 / (1 + np.exp(-z))), "sigmoid incorrect")
    with np.errstate(over="raise"):
        try:
            s = m.sigmoid(np.array([-1000.0, 1000.0]))
        except FloatingPointError:
            raise AssertionError("sigmoid(-1000) fait un overflow : rends-la stable (voir le sujet)")
    check(close(s, [0, 1]), "sigmoid([-1000, 1000]) doit valoir [0, 1]")


def ex01(d):
    no_loops(d, "logistic.py")
    m = load(d, "logistic.py")
    need(m, "predict_proba", "predict")
    X = np.array([[0.0, 0], [10, 0], [-10, 0]])
    theta = np.array([0.0, 1, 0])
    check(close(m.predict_proba(X, theta), [0.5, 1 / (1 + np.exp(-10)), 1 / (1 + np.exp(10))]), "predict_proba incorrect")
    p = m.predict(X, np.array([-0.1, 1, 0]))
    check(np.array_equal(p, [0, 1, 0]), "predict incorrect")
    check(np.issubdtype(np.asarray(p).dtype, np.integer), "predict doit renvoyer des entiers (0 ou 1)")
    check(np.array_equal(m.predict(X, theta, threshold=0.4), [1, 1, 0]), "le paramètre threshold n'est pas pris en compte")


def ex02(d):
    no_loops(d, "cross_entropy.py")
    m = load(d, "cross_entropy.py")
    need(m, "bce", "bce_gradient")
    y = np.array([1.0, 0, 1])
    p = np.array([0.9, 0.2, 0.6])
    check(close(m.bce(p, y), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "bce incorrect")
    check(np.isfinite(m.bce(np.array([1.0, 0.0]), np.array([0.0, 1.0]))), "bce doit rester finie si p vaut 0 ou 1 (clip !)")
    X, y = _data()
    theta = np.array([0.1, -0.3, 0.2])
    Xb = np.c_[np.ones(len(X)), X]
    loss = lambda: m.bce(1 / (1 + np.exp(-(Xb @ theta))), y)
    grad_check(loss, theta, m.bce_gradient(X, y, theta.copy()), "bce_gradient")


def ex03(d):
    m = load(d, "train.py")
    need(m, "fit")
    X, y = _data(seed=3)
    theta, history = m.fit(X, y, lr=0.5, n_iter=1000)
    check(len(history) == 1000 and history[-1] < history[0], "history incorrect ou la loss ne diminue pas")
    Xb = np.c_[np.ones(len(X)), X]
    acc = np.mean(((Xb @ theta) > 0) == y)
    check(acc > 0.95, f"précision trop faible après entraînement : {acc:.2f} (attendu > 0.95)")


def ex04(d):
    no_loops(d, "softmax.py")
    m = load(d, "softmax.py")
    need(m, "softmax", "cross_entropy", "softmax_ce_gradient")
    Z = np.array([[1.0, 2, 3], [1000, 1000, 1000]])
    with np.errstate(over="raise", invalid="raise"):
        try:
            P = m.softmax(Z)
        except FloatingPointError:
            raise AssertionError("softmax fait un overflow : soustrais le max de chaque ligne")
    e = np.exp([1.0, 2, 3])
    check(close(P[0], e / e.sum()) and close(P[1], 1 / 3), "softmax incorrect")
    check(close(P.sum(axis=1), 1), "chaque ligne de softmax doit sommer à 1")
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(6, 4))
    y = np.array([0, 3, 1, 1, 2, 0])
    Pz = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    check(close(m.cross_entropy(Pz, y), -np.mean(np.log(Pz[np.arange(6), y]))), "cross_entropy incorrect")
    grad_check(lambda: m.cross_entropy(m.softmax(Z), y), Z, m.softmax_ce_gradient(Z.copy(), y), "softmax_ce_gradient")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03), ("ex04", ex04)]
