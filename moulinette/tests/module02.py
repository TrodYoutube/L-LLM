import math

import numpy as np

from _utils import check, close, load, need, num_derivative


def ex00(d):
    m = load(d, "derivative.py")
    need(m, "derivative")
    check(abs(m.derivative(lambda x: x**2, 3.0) - 6) < 1e-4, "dérivée de x² en 3 : attendu 6")
    check(abs(m.derivative(math.sin, 0.0) - 1) < 1e-4, "dérivée de sin en 0 : attendu 1")
    check(abs(m.derivative(lambda x: x**3, 2.0) - 12) < 1e-4, "dérivée de x³ en 2 : attendu 12")
    # La différence centrée est bien plus précise que (f(x+h) - f(x)) / h
    check(abs(m.derivative(math.exp, 1.0, h=1e-3) - math.e) < 1e-5, "utilise la différence CENTRÉE (voir le sujet)")


def ex01(d):
    m = load(d, "gradient.py")
    need(m, "gradient")
    f = lambda v: v[0] ** 2 + 3 * v[1] ** 2 + v[0] * v[2]
    x = np.array([1.0, 2.0, 3.0])
    g = m.gradient(f, x)
    check(isinstance(g, np.ndarray) and g.shape == (3,), "gradient doit renvoyer un np.ndarray de même forme que x")
    check(close(g, [2 * 1 + 3, 6 * 2, 1], tol=1e-4), f"gradient incorrect : reçu {g}")
    check(np.array_equal(x, [1.0, 2.0, 3.0]), "gradient ne doit pas modifier x")


def ex02(d):
    m = load(d, "descent.py")
    need(m, "gradient_descent")
    x, hist = m.gradient_descent(lambda x: 2 * (x - 3), 0.0, lr=0.1, n_iter=100)
    check(abs(x - 3) < 1e-4, f"minimum de (x-3)² : attendu 3, reçu {x}")
    check(len(hist) == 101, "history doit contenir x0 + une valeur par itération (n_iter + 1)")
    check(hist[0] == 0.0 and abs(hist[1] - 0.6) < 1e-9, "history[1] doit valoir x0 - lr * grad(x0) = 0.6")
    grad = lambda v: np.array([2 * (v[0] - 1), 4 * (v[1] + 2)])
    x, _ = m.gradient_descent(grad, np.array([5.0, 5.0]), lr=0.1, n_iter=200)
    check(close(x, [1, -2], tol=1e-4), "gradient_descent doit marcher en plusieurs dimensions (np.array)")
    x, _ = m.gradient_descent(lambda x: 2 * x, 1.0, lr=1.1, n_iter=10)
    check(abs(x) > 1, "avec un learning rate trop grand, la descente doit diverger (ne triche pas)")


def ex03(d):
    m = load(d, "chain_rule.py")
    need(m, "sigmoid", "sigmoid_prime", "relu", "relu_prime", "f", "f_prime", "neuron", "neuron_grad")
    xs = np.linspace(-3, 3, 13)
    check(close(m.sigmoid(0.0), 0.5), "sigmoid(0) doit valoir 0.5")
    check(close(m.sigmoid(xs), 1 / (1 + np.exp(-xs))), "sigmoid doit marcher sur des np.array")
    check(close(m.sigmoid_prime(xs), num_derivative(m.sigmoid, xs), tol=1e-5), "sigmoid_prime incorrect")
    check(np.array_equal(m.relu(np.array([-2.0, 0, 3])), [0, 0, 3]), "relu incorrect")
    check(np.array_equal(m.relu_prime(np.array([-2.0, 0, 3])), [0, 0, 1]), "relu_prime incorrect (0 pour x <= 0)")
    check(close(m.f(xs), 1 / (1 + np.exp(-(3 * xs**2 + 1)))), "f(x) = sigmoid(3x² + 1) incorrect")
    check(close(m.f_prime(xs), num_derivative(m.f, xs), tol=1e-5), "f_prime incorrect : applique la règle de la chaîne")
    x, w, b = 1.5, -0.7, 0.3
    dw, db = m.neuron_grad(x, w, b)
    check(abs(dw - num_derivative(lambda w_: m.neuron(x, w_, b), w)) < 1e-6, "neuron_grad : d/dw incorrect")
    check(abs(db - num_derivative(lambda b_: m.neuron(x, w, b_), b)) < 1e-6, "neuron_grad : d/db incorrect")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03)]
