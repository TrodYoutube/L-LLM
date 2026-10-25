import os
import subprocess
import sys
import tempfile

from _utils import check, forbidden_imports, forbidden_names, load, need, no_for_statements, raises


def ex00(d):
    path = os.path.join(d, "hello.py")
    check(os.path.isfile(path), "fichier manquant : hello.py")
    out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=5).stdout
    check(out == "Hello, IA!\n", f"sortie attendue 'Hello, IA!\\n', reçu {out!r}")


def ex01(d):
    forbidden_names(d, "basics.py", {"sum", "max", "min", "Counter", "sorted"})
    m = load(d, "basics.py")
    need(m, "ft_sum", "ft_max", "count_chars")
    check(m.ft_sum([1, 2, 3, 4]) == 10, "ft_sum([1, 2, 3, 4]) doit valoir 10")
    check(m.ft_sum([]) == 0, "ft_sum([]) doit valoir 0")
    check(abs(m.ft_sum([0.5, 0.25]) - 0.75) < 1e-9, "ft_sum doit marcher avec des floats")
    check(m.ft_max([3, -1, 42, 7]) == 42, "ft_max([3, -1, 42, 7]) doit valoir 42")
    check(m.ft_max([-5, -2, -9]) == -2, "ft_max([-5, -2, -9]) doit valoir -2 (attention à l'init !)")
    check(raises(ValueError, m.ft_max, []), "ft_max([]) doit lever une ValueError")
    check(m.count_chars("hello") == {"h": 1, "e": 1, "l": 2, "o": 1}, "count_chars('hello') incorrect")
    check(m.count_chars("") == {}, "count_chars('') doit renvoyer {}")


def ex02(d):
    no_for_statements(d, "comprehensions.py")
    m = load(d, "comprehensions.py")
    need(m, "squares", "evens", "flatten", "word_lengths")
    check(m.squares(5) == [0, 1, 4, 9, 16], "squares(5) doit renvoyer [0, 1, 4, 9, 16]")
    check(m.squares(0) == [], "squares(0) doit renvoyer []")
    check(m.evens([1, 2, 3, 4, 5, 6, -2]) == [2, 4, 6, -2], "evens() incorrect")
    check(m.flatten([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5], "flatten() incorrect")
    check(
        m.word_lengths("le chat dort") == {"le": 2, "chat": 4, "dort": 4},
        "word_lengths('le chat dort') incorrect",
    )


def ex03(d):
    forbidden_imports(d, "matrix.py", {"numpy"})
    m = load(d, "matrix.py")
    need(m, "shape", "mat_add", "transpose", "dot", "mat_mul")
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    check(m.shape(A) == (2, 3), "shape(A) doit renvoyer le tuple (2, 3)")
    check(m.mat_add(A, A) == [[2, 4, 6], [8, 10, 12]], "mat_add incorrect")
    check(raises(ValueError, m.mat_add, A, B), "mat_add avec des tailles différentes doit lever ValueError")
    check(m.transpose(A) == [[1, 4], [2, 5], [3, 6]], "transpose incorrect")
    check(m.dot([1, 2, 3], [4, 5, 6]) == 32, "dot([1,2,3], [4,5,6]) doit valoir 32")
    check(raises(ValueError, m.dot, [1, 2], [1, 2, 3]), "dot avec des tailles différentes doit lever ValueError")
    check(m.mat_mul(A, B) == [[58, 64], [139, 154]], "mat_mul(A, B) incorrect")
    check(m.mat_mul([[1, 0], [0, 1]], [[5], [6]]) == [[5], [6]], "mat_mul avec une matrice colonne incorrect")
    check(raises(ValueError, m.mat_mul, A, A), "mat_mul (2,3)x(2,3) doit lever ValueError")
    check(A == [[1, 2, 3], [4, 5, 6]], "tes fonctions ne doivent pas modifier leurs arguments")


def ex04(d):
    m = load(d, "vector.py")
    need(m, "Vector")
    V = m.Vector
    u, v = V([1, 2, 3]), V([4, 5, 6])
    check(len(u) == 3, "len(Vector([1,2,3])) doit valoir 3")
    check(u[1] == 2.0, "Vector([1,2,3])[1] doit valoir 2.0")
    check(isinstance(u[0], float), "les valeurs doivent être stockées en float")
    check(u + v == V([5, 7, 9]), "u + v incorrect (ou __eq__ manquant)")
    check(v - u == V([3, 3, 3]), "v - u incorrect")
    check(u * 2 == V([2, 4, 6]), "u * 2 incorrect")
    check(2 * u == V([2, 4, 6]), "2 * u incorrect (pense à __rmul__)")
    check(u.dot(v) == 32.0, "u.dot(v) doit valoir 32.0")
    check(abs(V([3, 4]).norm() - 5.0) < 1e-9, "Vector([3, 4]).norm() doit valoir 5.0")
    check(raises(ValueError, lambda: u + V([1, 2])), "additionner des vecteurs de tailles différentes doit lever ValueError")
    check(repr(V([1, 2])) == "Vector([1.0, 2.0])", f"repr incorrect : {repr(V([1, 2]))!r}")
    check(u == V([1, 2, 3]) and not (u == v), "__eq__ incorrect")
    w = u + v
    check(u == V([1, 2, 3]), "u + v ne doit pas modifier u")
    check(w is not u, "u + v doit renvoyer un NOUVEAU Vector")


def ex05(d):
    forbidden_imports(d, "stats.py", {"numpy", "statistics", "pandas"})
    m = load(d, "stats.py")
    need(m, "mean", "variance", "std", "standardize", "load_column")
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    check(m.mean(data) == 5.0, "mean incorrect")
    check(m.variance(data) == 4.0, "variance incorrecte (variance de population : divise par n)")
    check(m.std(data) == 2.0, "std incorrect")
    z = m.standardize(data)
    check(abs(m.mean(z)) < 1e-9 and abs(m.std(z) - 1) < 1e-9, "standardize : le résultat doit avoir moyenne 0 et std 1")
    check(raises(ValueError, m.mean, []), "mean([]) doit lever ValueError")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("surface,prix\n30,150.5\n45,200\n60,260.25\n")
        path = f.name
    try:
        check(m.load_column(path, "prix") == [150.5, 200.0, 260.25], "load_column(path, 'prix') incorrect")
        check(m.load_column(path, "surface") == [30.0, 45.0, 60.0], "load_column(path, 'surface') incorrect")
        check(raises(KeyError, m.load_column, path, "nope"), "load_column avec une colonne inconnue doit lever KeyError")
    finally:
        os.unlink(path)


EXERCISES = [
    ("ex00", ex00),
    ("ex01", ex01),
    ("ex02", ex02),
    ("ex03", ex03),
    ("ex04", ex04),
    ("ex05", ex05),
]
