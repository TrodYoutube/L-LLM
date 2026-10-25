import numpy as np

from _utils import check, close, load, need, no_loops


def _data(seed=0, m=200, n=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, n))
    theta = rng.normal(size=n + 1)
    y = theta[0] + X @ theta[1:] + rng.normal(scale=0.01, size=m)
    return X, y, theta


def ex00(d):
    no_loops(d, "predict.py")
    m = load(d, "predict.py")
    need(m, "predict")
    X = np.array([[1.0, 2], [3, 4], [0, 0]])
    theta = np.array([10.0, 1, -1])
    p = m.predict(X, theta)
    check(np.shape(p) == (3,), f"predict doit renvoyer un vecteur de forme (m,), reçu {np.shape(p)}")
    check(close(p, [9, 9, 10]), "predict incorrect : theta[0] est le biais")
    check(close(m.predict(np.array([[50.0]]), np.array([40.0, 3.5])), [215]), "predict incorrect en dimension 1")


def ex01(d):
    no_loops(d, "loss.py")
    m = load(d, "loss.py")
    need(m, "mse", "r2_score")
    y = np.array([1.0, 2, 3, 4])
    check(close(m.mse(y, y), 0), "mse(y, y) doit valoir 0")
    check(close(m.mse(np.array([2.0, 2, 2, 2]), y), 1.5), "mse incorrect")
    check(close(m.r2_score(y, y), 1), "r2_score(y, y) doit valoir 1")
    check(close(m.r2_score(np.full(4, y.mean()), y), 0), "r2_score d'une prédiction constante = moyenne doit valoir 0")


def ex02(d):
    no_loops(d, "gradient.py")
    m = load(d, "gradient.py")
    need(m, "gradient")
    X, y, _ = _data()
    theta = np.array([0.5, -1, 2, 0.1])
    g = m.gradient(X, y, theta)
    check(np.shape(g) == (4,), "gradient doit avoir la même forme que theta")
    Xb = np.c_[np.ones(len(X)), X]
    expected = 2 / len(X) * Xb.T @ (Xb @ theta - y)
    check(close(g, expected), "gradient incorrect (rappel : J = moyenne des erreurs², donc facteur 2/m)")


def ex03(d):
    m = load(d, "train.py")
    need(m, "fit")
    X, y, theta_true = _data(seed=1)
    theta, history = m.fit(X, y, lr=0.1, n_iter=500)
    check(np.shape(theta) == (4,), "fit doit renvoyer theta de forme (n+1,)")
    check(len(history) == 500, "history doit contenir une loss par itération")
    check(history[-1] < history[0], "la loss doit diminuer pendant l'entraînement")
    check(close(theta, theta_true, tol=0.02), f"fit n'a pas convergé vers les bons paramètres : {theta} vs {theta_true}")


def ex04(d):
    no_loops(d, "normal_equation.py")
    m = load(d, "normal_equation.py")
    need(m, "normal_equation")
    X, y, theta_true = _data(seed=2)
    theta = m.normal_equation(X, y)
    Xb = np.c_[np.ones(len(X)), X]
    check(close(theta, np.linalg.lstsq(Xb, y, rcond=None)[0], tol=1e-6), "normal_equation incorrect")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03), ("ex04", ex04)]
