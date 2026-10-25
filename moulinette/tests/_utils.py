"""Outils partagés par les tests de la moulinette."""

import ast
import importlib.util
import os
import sys


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def load(ex_dir, filename):
    """Importe le fichier rendu par l'étudiant et renvoie le module."""
    path = os.path.join(ex_dir, filename)
    check(os.path.isfile(path), f"fichier manquant : {filename}")
    name = os.path.splitext(filename)[0]
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def need(module, *names):
    for n in names:
        check(hasattr(module, n), f"`{n}` introuvable dans {module.__name__}.py")


def close(a, b, tol=1e-6):
    import numpy as np

    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=tol)


def _tree(ex_dir, filename):
    with open(os.path.join(ex_dir, filename)) as f:
        return ast.parse(f.read())


def no_loops(ex_dir, filename, allow_comprehensions=False):
    """Interdit for/while (et les compréhensions si demandé)."""
    banned = (ast.For, ast.While, ast.AsyncFor)
    if not allow_comprehensions:
        banned += (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for node in ast.walk(_tree(ex_dir, filename)):
        if isinstance(node, banned):
            raise AssertionError(
                f"{filename}:{node.lineno} : boucle interdite dans cet exercice, vectorise avec NumPy !"
            )


def no_for_statements(ex_dir, filename):
    """Interdit for/while mais autorise les compréhensions."""
    no_loops(ex_dir, filename, allow_comprehensions=True)


def forbidden_names(ex_dir, filename, names):
    for node in ast.walk(_tree(ex_dir, filename)):
        ident = None
        if isinstance(node, ast.Name):
            ident = node.id
        elif isinstance(node, ast.Attribute):
            ident = node.attr
        if ident in names:
            raise AssertionError(f"{filename}:{node.lineno} : `{ident}` est interdit dans cet exercice")


def forbidden_imports(ex_dir, filename, modules):
    for node in ast.walk(_tree(ex_dir, filename)):
        found = []
        if isinstance(node, ast.Import):
            found = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found = [node.module.split(".")[0]]
        for m in found:
            if m in modules:
                raise AssertionError(f"{filename}:{node.lineno} : import de `{m}` interdit dans cet exercice")


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False


def num_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def grad_check(f, x, analytic, name, h=1e-5, tol=1e-4):
    """Compare un gradient analytique avec un gradient numérique (x modifié en place)."""
    import numpy as np

    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        num[i] = (fp - fm) / (2 * h)
    err = np.max(np.abs(num - analytic)) / max(1e-8, np.max(np.abs(num)) + np.max(np.abs(analytic)))
    check(err < tol, f"gradient {name} faux (erreur relative {err:.2e}) : vérifie ton backward")
