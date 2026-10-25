import math
import random

from _utils import check, load, need


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def ex00(d):
    m = load(d, "value.py")
    need(m, "Value")
    V = m.Value
    a, b = V(2.0), V(-3.0)
    c = a * b + 10
    check(isinstance(c, V), "a * b + 10 doit renvoyer une Value")
    check(_close(c.data, 4.0), f"(2 * -3) + 10 doit valoir 4, reçu {c.data}")
    check(c._op == "+", "le noeud final de a * b + 10 doit avoir _op == '+'")
    check(len(c._prev) == 2, "c._prev doit contenir les 2 noeuds parents")
    check(any(p._op == "*" for p in c._prev), "un des parents de c doit être le noeud '*'")
    check(_close((3 + a).data, 5) and _close((3 * a).data, 6), "pense à __radd__ et __rmul__ (3 + a, 3 * a)")
    check(_close((a - b).data, 5) and _close((-a).data, -2), "soustraction ou négation incorrecte")
    check(_close((1 - a).data, -1), "pense à __rsub__ (1 - a)")
    check(repr(V(1.5)) == "Value(data=1.5)", f"repr incorrect : {repr(V(1.5))!r}")


def ex01(d):
    m = load(d, "value.py")
    need(m, "Value")
    V = m.Value
    check(_close(V(1.0).grad, 0.0), "grad doit valoir 0.0 à la création")
    a, b, c = V(2.0), V(-3.0), V(10.0)
    e = a * b
    L = e + c
    L.backward()
    check(_close(L.grad, 1.0), "backward() doit mettre le grad du noeud racine à 1")
    check(_close(a.grad, -3) and _close(b.grad, 2) and _close(c.grad, 1), "gradients faux sur L = a*b + c")
    x = V(3.0)
    y = x * x + x
    y.backward()
    check(_close(x.grad, 7.0), f"y = x*x + x : dy/dx doit valoir 7, reçu {x.grad} (les grads doivent s'ADDITIONNER : +=)")
    x = V(-2.0)
    z = (x - 1) * (x + 4) - x * 3
    z.backward()
    check(_close(x.grad, 2 * -2 + 3 - 3), "gradient faux sur (x-1)(x+4) - 3x")


def ex02(d):
    m = load(d, "value.py")
    need(m, "Value")
    V = m.Value

    def f(xv):
        x = V(xv)
        y = ((x**2 + 1).tanh() * x.exp() / (x + 3) + (x * 2 - 1).relu() + 1 / (x + 4)) ** 1.5
        return x, y

    for xv in (0.3, 1.2, 2.0):
        x, y = f(xv)
        y.backward()
        num = (f(xv + 1e-6)[1].data - f(xv - 1e-6)[1].data) / 2e-6
        check(_close(x.grad, num, 1e-4), f"gradient faux pour x={xv} : {x.grad} vs {num} (vérifie pow / div / tanh / exp / relu)")
    check(_close(V(-1.0).relu().data, 0) and _close(V(2.5).relu().data, 2.5), "relu incorrect")
    check(_close(V(0.5).tanh().data, math.tanh(0.5)), "tanh incorrect")


def ex03(d):
    load(d, "value.py")
    m = load(d, "nn.py")
    need(m, "Neuron", "Layer", "MLP")
    random.seed(1)
    n = m.Neuron(3)
    check(len(n.parameters()) == 4, "Neuron(3).parameters() doit contenir 3 poids + 1 biais")
    out = n([1.0, -2.0, 0.5])
    check(hasattr(out, "backward"), "appeler un Neuron doit renvoyer une Value")
    lay = m.Layer(3, 4)
    outs = lay([1.0, 2.0, 3.0])
    check(isinstance(outs, list) and len(outs) == 4, "Layer(3, 4)(x) doit renvoyer une liste de 4 Values")
    net = m.MLP(3, [4, 4, 1])
    check(len(net.parameters()) == 41, f"MLP(3, [4, 4, 1]) doit avoir 41 paramètres, reçu {len(net.parameters())}")
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    for attempt in range(3):
        random.seed(attempt)
        net = m.MLP(3, [4, 4, 1])
        losses = []
        for _ in range(200):
            preds = [net(x) for x in xs]
            preds = [p[0] if isinstance(p, list) else p for p in preds]
            loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), m.Value(0.0) if hasattr(m, "Value") else 0.0)
            net.zero_grad()
            loss.backward()
            for p in net.parameters():
                p.data -= 0.05 * p.grad
            losses.append(loss.data)
        if losses[-1] < 0.05:
            return
    raise AssertionError(f"ton MLP n'apprend pas : loss finale {losses[-1]:.3f} (attendu < 0.05). zero_grad marche ?")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03)]
