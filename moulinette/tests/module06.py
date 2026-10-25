import gzip
import os
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

from _utils import check, close, grad_check, load, need, raises

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_idx(path, magic, header, payload):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">" + "I" * (1 + len(header)), magic, *header))
        f.write(payload.astype(np.uint8).tobytes())


def ex00(d):
    m = load(d, "mnist.py")
    need(m, "load_images", "load_labels")
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(5, 4, 3))
    labels = np.array([3, 1, 4, 1, 5])
    with tempfile.TemporaryDirectory() as tmp:
        pi, pl, bad = (os.path.join(tmp, n) for n in ("img.gz", "lab.gz", "bad.gz"))
        _write_idx(pi, 2051, [5, 4, 3], imgs)
        _write_idx(pl, 2049, [5], labels)
        _write_idx(bad, 1234, [5], labels)
        X = m.load_images(pi)
        check(X.shape == (5, 12), f"load_images : forme attendue (5, 12), reçu {X.shape}")
        check(X.dtype == np.float32, "load_images doit renvoyer des float32")
        check(close(X, imgs.reshape(5, 12) / 255), "load_images : les pixels doivent être divisés par 255")
        y = m.load_labels(pl)
        check(np.array_equal(y, labels) and np.issubdtype(y.dtype, np.integer), "load_labels incorrect")
        check(raises(ValueError, m.load_labels, bad), "un magic number invalide doit lever ValueError")
        check(raises(ValueError, m.load_images, pl), "load_images sur un fichier de labels doit lever ValueError")


def ex01(d):
    m = load(d, "layers.py")
    need(m, "Dense", "ReLU")
    rng = np.random.default_rng(0)
    layer = m.Dense(5, 3, rng=np.random.default_rng(1))
    check(layer.W.shape == (5, 3) and layer.b.shape == (3,), "Dense(5, 3) : W doit être (5, 3) et b (3,)")
    check(np.all(layer.b == 0), "b doit être initialisé à 0")
    big = m.Dense(1000, 200, rng=np.random.default_rng(2))
    check(abs(big.W.std() - np.sqrt(2 / 1000)) < 0.003, "initialise W avec He : normale * sqrt(2 / n_in)")
    X = rng.normal(size=(4, 5))
    out = layer.forward(X)
    check(close(out, X @ layer.W + layer.b), "Dense.forward incorrect")
    G = rng.normal(size=(4, 3))
    dX = layer.backward(G)
    loss = lambda: np.sum(layer.forward(X) * G)
    grad_check(loss, layer.W, layer.dW.copy(), "dW")
    grad_check(loss, layer.b, layer.db.copy(), "db")
    grad_check(loss, X, dX, "dX")
    relu = m.ReLU()
    Z = np.array([[-1.0, 2], [3, -4]])
    check(np.array_equal(relu.forward(Z), [[0, 2], [3, 0]]), "ReLU.forward incorrect")
    check(np.array_equal(relu.backward(np.ones((2, 2))), [[0, 1], [1, 0]]), "ReLU.backward incorrect")


def ex02(d):
    m = load(d, "loss.py")
    need(m, "SoftmaxCrossEntropy")
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 5))
    y = rng.integers(0, 5, size=8)
    crit = m.SoftmaxCrossEntropy()
    loss = crit.forward(Z, y)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    check(close(loss, -np.mean(np.log(P[np.arange(8), y]))), "SoftmaxCrossEntropy.forward incorrect")
    dZ = crit.backward()
    check(np.shape(dZ) == Z.shape, "backward doit renvoyer un gradient de la forme des logits")
    grad_check(lambda: m.SoftmaxCrossEntropy().forward(Z, y), Z, dZ, "dlogits")
    check(np.isfinite(crit.forward(np.array([[1000.0, -1000.0]]), np.array([1]))), "forward doit être stable numériquement")


def _spiral(n=100, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(scale=0.2, size=n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n, c))
    return np.concatenate(X), np.concatenate(y)


def ex03(d):
    for f in ("layers.py", "loss.py"):
        load(d, f)
    m = load(d, "network.py")
    need(m, "Network")
    layers_mod = sys.modules["layers"]
    loss_mod = sys.modules["loss"]
    X, y = _spiral()
    rng = np.random.default_rng(0)
    net = m.Network(
        [layers_mod.Dense(2, 64, rng=rng), layers_mod.ReLU(), layers_mod.Dense(64, 3, rng=rng)],
        loss_mod.SoftmaxCrossEntropy(),
    )
    first = net.train_step(X, y, lr=0.5)
    for _ in range(1500):
        last = net.train_step(X, y, lr=0.5)
    check(last < first, "train_step doit renvoyer la loss, et elle doit diminuer")
    pred = net.predict(X)
    check(np.shape(pred) == (300,), "predict doit renvoyer un vecteur de classes (entiers)")
    acc = np.mean(pred == y)
    check(acc > 0.9, f"précision sur la spirale : {acc:.2f}, attendu > 0.90")


def ex04(d):
    for f in ("mnist.py", "layers.py", "loss.py", "network.py", "train_mnist.py"):
        check(os.path.isfile(os.path.join(d, f)), f"fichier manquant : {f}")
    check(os.path.isdir(os.path.join(ROOT, "data", "mnist")), "MNIST introuvable : lance d'abord ./data/get_data.sh")
    proc = subprocess.run([sys.executable, "train_mnist.py"], cwd=d, capture_output=True, text=True, timeout=1500)
    check(proc.returncode == 0, f"train_mnist.py a planté : {(proc.stderr.strip().splitlines() or [''])[-1]}")
    lines = proc.stdout.strip().splitlines()
    match = re.match(r"test accuracy: ([0-9.]+)$", lines[-1] if lines else "")
    check(match, "la dernière ligne affichée doit être 'test accuracy: 0.XXXX'")
    acc = float(match.group(1))
    check(acc >= 0.97, f"précision sur le test set : {acc:.4f}, attendu >= 0.97")


EXERCISES = [
    ("ex00", ex00),
    ("ex01", ex01),
    ("ex02", ex02),
    ("ex03", ex03),
    ("ex04", ex04, {"long": True, "timeout": 1500}),
]
