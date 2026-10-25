import os
import re
import subprocess
import sys

import numpy as np

from _utils import check, close, load, need


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        raise AssertionError("PyTorch n'est pas installé : voir la section Installation du sujet")


def ex00(d):
    torch = _torch()
    m = load(d, "tensors.py")
    need(m, "to_tensor", "derivative", "neuron_grad")
    t = m.to_tensor(np.array([[1, 2], [3, 4]]))
    check(isinstance(t, torch.Tensor), "to_tensor doit renvoyer un torch.Tensor")
    check(t.dtype == torch.float32 and t.shape == (2, 2), "to_tensor doit renvoyer un tenseur float32 de même forme")
    check(abs(m.derivative(lambda x: x**3 + 2 * x, 2.0) - 14) < 1e-5, "derivative(x³ + 2x, 2) doit valoir 14")
    check(isinstance(m.derivative(lambda x: x * 2, 1.0), float), "derivative doit renvoyer un float Python (.item())")
    x, w, b = 1.5, -0.7, 0.3
    s = 1 / (1 + np.exp(-(w * x + b)))
    dw, db = m.neuron_grad(x, w, b)
    check(abs(dw - s * (1 - s) * x) < 1e-5 and abs(db - s * (1 - s)) < 1e-5, "neuron_grad incorrect")


def ex01(d):
    torch = _torch()
    m = load(d, "mlp.py")
    need(m, "MLP", "train_one_epoch", "evaluate")
    model = m.MLP([784, 128, 10])
    check(isinstance(model, torch.nn.Module), "MLP doit hériter de nn.Module")
    n_params = sum(p.numel() for p in model.parameters())
    check(n_params == 784 * 128 + 128 + 128 * 10 + 10, f"MLP([784, 128, 10]) : nombre de paramètres incorrect ({n_params})")
    check(model(torch.zeros(7, 784)).shape == (7, 10), "la sortie de MLP doit être de forme (batch, 10)")
    torch.manual_seed(0)
    X = torch.randn(600, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).long()
    model = m.MLP([2, 32, 32, 2])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = [m.train_one_epoch(model, X, y, opt, batch_size=32) for _ in range(30)]
    check(all(isinstance(l, float) for l in losses), "train_one_epoch doit renvoyer la loss moyenne (float)")
    check(losses[-1] < losses[0], "la loss doit diminuer")
    acc = m.evaluate(model, X, y)
    check(isinstance(acc, float) and acc > 0.95, f"evaluate : précision {acc} sur un XOR (attendu float > 0.95)")


def ex02(d):
    torch = _torch()
    m = load(d, "cnn.py")
    need(m, "CNN")
    model = m.CNN()
    out = model(torch.zeros(4, 1, 28, 28))
    check(out.shape == (4, 10), "CNN doit prendre (N, 1, 28, 28) et renvoyer (N, 10)")
    check(any(isinstance(mod, torch.nn.Conv2d) for mod in model.modules()), "ton CNN doit utiliser au moins un nn.Conv2d")
    n_params = sum(p.numel() for p in model.parameters())
    check(n_params < 500_000, f"trop de paramètres ({n_params}), reste sous 500k")
    proc = subprocess.run([sys.executable, "cnn.py"], cwd=d, capture_output=True, text=True, timeout=1500)
    check(proc.returncode == 0, f"cnn.py a planté : {(proc.stderr.strip().splitlines() or [''])[-1]}")
    lines = proc.stdout.strip().splitlines()
    match = re.match(r"test accuracy: ([0-9.]+)$", lines[-1] if lines else "")
    check(match, "la dernière ligne affichée doit être 'test accuracy: 0.XXXX'")
    check(float(match.group(1)) >= 0.99, f"précision {match.group(1)}, attendu >= 0.99")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02, {"long": True, "timeout": 1500})]
