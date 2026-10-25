# Module 01 — NumPy

> *« Si tu écris une boucle `for` sur une matrice, tu as perdu. »*

**Objectif :** manipuler des tableaux de nombres avec NumPy, et surtout **penser en vecteurs** plutôt qu'en boucles.
**Durée estimée :** 2 à 3 jours.
**Rendu :** `rendu/module01/exXX/`

---

## Le cours

### Pourquoi NumPy ?

Ton `mat_mul` du module 00 marche, mais il est **lent** : Python interprète chaque tour de boucle. NumPy, c'est une bibliothèque écrite en **C** (et en Fortran) qui fait les boucles pour toi, sur des tableaux contigus en mémoire, comme tes `double *` en C. Résultat : 100 à 1000 fois plus rapide.

```python
import numpy as np    # la convention universelle, écris-la toujours comme ça
```

### Le `ndarray`

```python
a = np.array([1, 2, 3])               # vecteur, shape (3,)
M = np.array([[1, 2, 3],
              [4, 5, 6]])             # matrice, shape (2, 3)

M.shape    # (2, 3)  ← LA propriété la plus importante. Affiche-la tout le temps.
M.ndim     # 2       (nombre de dimensions)
M.size     # 6       (nombre total d'éléments)
M.dtype    # int64   (type des éléments, comme en C : tous les éléments ont le même)
```

**Créer des tableaux :**

```python
np.zeros((2, 3))          # matrice 2x3 de 0.0
np.ones((2, 3))           # de 1.0
np.eye(3)                 # matrice identité 3x3
np.arange(6)              # [0 1 2 3 4 5], comme range
np.linspace(0, 1, 5)      # [0. 0.25 0.5 0.75 1.], 5 points entre 0 et 1
np.arange(6).reshape(2, 3)  # [[0 1 2], [3 4 5]], change la forme (mêmes données)
```

### Les opérations élément par élément

Les opérateurs s'appliquent à **chaque élément**, sans boucle :

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
a + b        # [11 22 33]
a * b        # [10 40 90]   ← ATTENTION : ce n'est PAS un produit matriciel
a ** 2       # [1 4 9]
np.sqrt(a)   # [1. 1.41 1.73]
np.exp(a)    # exponentielle de chaque élément
a > 1        # [False True True], un tableau de booléens
```

### Le produit matriciel : `@`

```python
A = np.arange(6).reshape(2, 3)    # (2, 3)
B = np.arange(12).reshape(3, 4)   # (3, 4)
C = A @ B                         # (2, 4) : ta fonction mat_mul en un caractère
```

**Le réflexe vital :** avant chaque `@`, vérifie les formes : `(n, k) @ (k, m) → (n, m)`. 90 % des bugs en IA sont des erreurs de shape.

### Les réductions et `axis`

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
M.sum()            # 21, tout
M.sum(axis=0)      # [5 7 9], on "écrase" l'axe 0 (les lignes) → une somme par COLONNE
M.sum(axis=1)      # [6 15], on écrase l'axe 1 (les colonnes) → une somme par LIGNE
M.mean(axis=0)     # moyenne de chaque colonne
M.max(axis=1)      # max de chaque ligne
np.argmax(M, axis=1)  # [2 2], l'INDICE du max de chaque ligne
```

Moyen mnémotechnique : `axis=0` fait disparaître la dimension 0. Une matrice `(2, 3)` réduite sur `axis=0` donne un vecteur `(3,)`.

Ajoute `keepdims=True` pour garder la dimension : `M.sum(axis=1, keepdims=True)` a la forme `(2, 1)` au lieu de `(2,)`. Très utile pour le broadcasting.

### Le broadcasting (le super-pouvoir)

Quand deux tableaux ont des formes différentes, NumPy « étire » virtuellement le plus petit, sans copier la mémoire :

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])       # (2, 3)
M + 10                          # 10 est étiré en (2, 3) → [[11 12 13], [14 15 16]]
M + np.array([100, 200, 300])   # (3,) étiré en (2, 3) : ajouté à CHAQUE LIGNE
M - M.mean(axis=0)              # centre chaque colonne. C'est le début de standardize !
```

**La règle :** on compare les formes **en partant de la droite**. Deux dimensions sont compatibles si elles sont égales ou si l'une vaut 1.

```
(2, 3) et (3,)    → (2, 3) et (1, 3) → OK, résultat (2, 3)
(2, 3) et (2,)    → (2, 3) et (1, 2) → 3 ≠ 2, ERREUR
(2, 3) et (2, 1)  → OK, résultat (2, 3) : chaque ligne reçoit sa propre valeur
(5, 1, 3) et (1, 7, 3) → OK, résultat (5, 7, 3)
```

Pour ajouter une dimension de taille 1 : `v[:, None]` (ou `v[:, np.newaxis]`) transforme un `(n,)` en `(n, 1)`, et `v[None, :]` le transforme en `(1, n)`.

### L'indexation

```python
M = np.arange(12).reshape(3, 4)
M[0, 1]        # élément ligne 0, colonne 1
M[1]           # ligne 1 entière, shape (4,)
M[:, 2]        # colonne 2 entière, shape (3,)
M[:2, 1:3]     # sous-matrice : lignes 0-1, colonnes 1-2

# Indexation par masque booléen
a = np.array([-1, 5, -3, 8])
a[a > 0]       # [5 8]

# Indexation par tableau d'indices ("fancy indexing")
a[[0, 3]]      # [-1 8]
M[[0, 2, 1], [1, 1, 3]]   # [M[0,1], M[2,1], M[1,3]], un élément par paire d'indices
np.eye(3)[[2, 0]]         # les lignes 2 puis 0 de l'identité. Pense au one-hot…
```

### Combiner des tableaux

```python
np.c_[a, b]                    # colle des colonnes côte à côte
np.concatenate([A, B], axis=0) # empile verticalement
np.hstack, np.vstack           # idem, en plus lisible
```

### L'aléatoire

```python
rng = np.random.default_rng(42)   # générateur avec une graine : résultats reproductibles
rng.normal(size=(3, 2))           # loi normale
rng.uniform(0, 1, size=5)
rng.permutation(10)               # une permutation aléatoire de 0..9
```

---

## Exercices

> **Règle du module : à partir de l'ex01, les boucles sont INTERDITES**, y compris les compréhensions de listes. La moulinette vérifie. Tout doit être vectorisé.

### ex00 — Premiers tableaux

| | |
|---|---|
| Fichier | `arrays.py` |

- `zeros(r, c)` : une matrice `(r, c)` de zéros.
- `identity(n)` : la matrice identité `n × n`.
- `arange_matrix(r, c)` : une matrice `(r, c)` contenant `0, 1, ..., r*c - 1` ligne par ligne.
- `describe(a)` : renvoie le tuple `(shape, ndim, size)`.

---

### ex01 — Vectorisation

| | |
|---|---|
| Fichier | `vectorize.py` |
| Interdit | toute boucle |

- `dot(u, v)` : le produit scalaire, **sans** `np.dot` ni `@` (utilise `*` et `np.sum`).
- `mat_mul(A, B)` : le produit matriciel (là, `@` est autorisé).
- `row_norms(X)` : un vecteur contenant la norme de **chaque ligne** de `X`. Pour `X` de forme `(n, d)`, le résultat a la forme `(n,)`.
- `normalize_rows(X)` : divise chaque ligne par sa norme, pour que chaque ligne ait une norme de 1. Attention au broadcasting : `(n, d) / (n,)` ne marche pas, mais `(n, d) / (n, 1)` oui.

---

### ex02 — Broadcasting

| | |
|---|---|
| Fichier | `broadcasting.py` |
| Interdit | toute boucle |

- `add_bias(X)` : ajoute une colonne de `1` **au début** de `X`. Une matrice `(m, n)` devient `(m, n+1)`. On s'en servira pour intégrer le biais d'un modèle dans un produit matriciel.
- `standardize(X)` : standardise **chaque colonne** de `X` (moyenne 0, écart-type 1). C'est ton `standardize` du module 00, en une ligne.
- `pairwise_sq_dist(A, B)` : `A` est `(n, d)`, `B` est `(m, d)`. Renvoie la matrice `D` de forme `(n, m)` où `D[i, j]` est la distance euclidienne **au carré** entre `A[i]` et `B[j]`.

**Indice pour `pairwise_sq_dist` :** que se passe-t-il si tu fais `A[:, None, :] - B[None, :, :]` ? Écris les formes sur une feuille : `(n, 1, d)` et `(1, m, d)`…

---

### ex03 — Indexation

| | |
|---|---|
| Fichier | `indexing.py` |
| Interdit | toute boucle |

- `positives(a)` : les éléments strictement positifs de `a`.
- `one_hot(labels, n_classes)` : transforme un vecteur d'entiers en matrice **one-hot**. `[0, 2, 1]` avec 3 classes donne :
  ```
  [[1, 0, 0],
   [0, 0, 1],
   [0, 1, 0]]
  ```
  C'est comme ça qu'on représente les classes en classification (module 04). Il existe une solution en une ligne très élégante, regarde le cours.
- `argmax_rows(X)` : l'indice du max de chaque ligne.
- `accuracy(y_pred, y_true)` : la proportion de prédictions correctes (entre 0 et 1).
- `shuffle_together(X, y, seed)` : mélange `X` et `y` **avec la même permutation**, pour que `X[i]` corresponde toujours à `y[i]`. Utilise exactement `np.random.default_rng(seed).permutation(len(X))`. Ne modifie pas `X` ni `y`, renvoie `(X_mélangé, y_mélangé)`.

---

```bash
python3 moulinette/moulinette.py module01
```

Ensuite, place aux maths : [module 02](module02.md).
