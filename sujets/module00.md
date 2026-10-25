# Module 00 - Python pour les codeurs C

> *" Python, c'est du pseudo-code qui s'execute. "*

**Objectif :** etre a l'aise avec la syntaxe Python pour ne plus jamais y penser dans la suite.
**Duree estimee :** 1 a 3 jours.
**Rendu :** `rendu/module00/exXX/`

---

## Le cours

### Python vs C : ce qui change

| C | Python |
|---|---|
| Compile (`cc main.c && ./a.out`) | Interprete (`python3 main.py`) |
| Typage statique : `int x = 3;` | Typage dynamique : `x = 3` (le type est porte par la valeur) |
| Blocs avec `{ }` | Blocs avec **l'indentation** (4 espaces). C'est obligatoire, pas du style |
| `malloc` / `free` | Rien. Un garbage collector libere la memoire tout seul |
| `char *`, `\0` | `str`, immuable, avec une vraie longueur : `len(s)` |
| Tableaux de taille fixe | Listes dynamiques : `l = [1, 2, 3]`, puis `l.append(4)` |
| `struct` | `class` (avec des methodes) |
| `NULL` | `None` |
| `&&`, `\|\|`, `!` | `and`, `or`, `not` |
| `i++` | `i += 1` (il n'y a pas de `++`) |

### Les bases

```python
# Variables : pas de declaration, pas de type
age = 21
pi = 3.14
nom = "Alice"          # str
actif = True           # bool (majuscule !)
rien = None

# Affichage : les f-strings, c'est le printf moderne
print(f"{nom} a {age} ans, pi vaut {pi:.2f}")

# Conditions
if age >= 18:
    print("majeur")
elif age > 12:
    print("ado")
else:
    print("enfant")

# Boucles : le for de Python parcourt une COLLECTION, pas un compteur
for i in range(5):           # 0, 1, 2, 3, 4  (comme for (i = 0; i < 5; i++))
    print(i)

for lettre in "abc":         # on peut iterer directement sur une chaine
    print(lettre)

while age > 0:
    age -= 1

# Fonctions
def ajoute(a, b=10):         # b a une valeur par defaut
    return a + b

ajoute(1)       # 11
ajoute(1, 2)    # 3
ajoute(b=5, a=1)  # 6, on peut nommer les arguments
```

### Les structures de donnees

```python
# LISTE : tableau dynamique, mutable, peut contenir n'importe quoi
l = [3, 1, 4]
l.append(1)        # [3, 1, 4, 1]
l[0]               # 3
l[-1]              # 1, le dernier element (indices negatifs = depuis la fin)
l[1:3]             # [1, 4], le "slicing" [debut:fin[
len(l)             # 4
4 in l             # True

# TUPLE : comme une liste mais immuable
point = (2, 3)
x, y = point       # "unpacking" : x = 2, y = 3

# DICTIONNAIRE : table de hachage cle -> valeur (ton ft_hashmap, gratuit)
ages = {"alice": 22, "bob": 19}
ages["charlie"] = 25
ages["alice"]                 # 22
ages.get("zoe", 0)            # 0 si la cle n'existe pas (pas d'exception)
for nom, age in ages.items():
    print(nom, age)

# SET : ensemble sans doublons
s = {1, 2, 2, 3}   # {1, 2, 3}
```

**Attention, piege de C-iste :** en Python, les variables sont des **references**.

```python
a = [1, 2, 3]
b = a          # b pointe sur LA MeME liste (comme deux pointeurs)
b.append(4)
print(a)       # [1, 2, 3, 4]  <- surprise !
c = a.copy()   # vraie copie
```

### Les comprehensions (tu vas les adorer)

Construire une liste en une ligne :

```python
carres = [x * x for x in range(5)]             # [0, 1, 4, 9, 16]
pairs = [x for x in range(10) if x % 2 == 0]   # [0, 2, 4, 6, 8]
d = {mot: len(mot) for mot in ["ia", "42"]}    # {'ia': 2, '42': 2}
```

a lire comme : " la liste des `x * x` pour chaque `x` dans `range(5)` ".

### Les exceptions (au lieu des codes de retour)

En C, tu renvoies `-1` ou `NULL` pour signaler une erreur. En Python, on **leve une exception** :

```python
def divise(a, b):
    if b == 0:
        raise ValueError("division par zero")
    return a / b

try:
    divise(1, 0)
except ValueError as e:
    print("erreur :", e)
```

### Les classes

Une classe, c'est une `struct` avec des fonctions dedans (les **methodes**). `self` est l'equivalent du pointeur vers la structure qu'on passerait en premier argument en C.

```python
class Point:
    def __init__(self, x, y):     # le "constructeur", appele par Point(1, 2)
        self.x = x                 # attribut
        self.y = y

    def norme(self):               # methode
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __add__(self, other):      # surcharge de l'operateur +
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):            # ce qu'affiche print(p)
        return f"Point({self.x}, {self.y})"

p = Point(3, 4) + Point(1, 1)
print(p, p.norme())               # Point(4, 5) 6.403...
```

Les methodes `__xxx__` (" dunder methods ") permettent de surcharger les operateurs : `__add__` pour `+`, `__mul__` pour `*`, `__len__` pour `len()`, `__getitem__` pour `[]`, `__eq__` pour `==`... **Retiens bien ca, on s'en sert enormement au module 05.**

### Les fichiers

```python
with open("data.csv") as f:      # with ferme le fichier tout seul (pas de close)
    header = f.readline()         # premiere ligne
    for line in f:                # le reste, ligne par ligne
        champs = line.strip().split(",")
```

### Tester ton code

Mets ce bloc en bas de chacun de tes fichiers. Il ne s'execute que si tu lances le fichier directement, pas quand la moulinette l'importe :

```python
if __name__ == "__main__":
    print(ft_sum([1, 2, 3]))   # tes tests ici
```

Pour experimenter, tape `python3` tout court : tu obtiens une console interactive. Et `help(list)` te donne la doc, c'est ton `man`.

---

## Exercices

### ex00 - Hello, IA!

| | |
|---|---|
| Fichier | `hello.py` |
| Interdit | rien |

Un programme qui affiche `Hello, IA!` suivi d'un retour a la ligne.

```
$ python3 hello.py
Hello, IA!
```

---

### ex01 - Les basiques

| | |
|---|---|
| Fichier | `basics.py` |
| Interdit | `sum`, `max`, `min`, `sorted`, `Counter` |

Recode ces fonctions avec des boucles, comme en C :

- `ft_sum(lst)` : renvoie la somme des elements (`0` pour une liste vide).
- `ft_max(lst)` : renvoie le plus grand element. Leve une `ValueError` si la liste est vide.
- `count_chars(s)` : renvoie un dictionnaire `{caractere: nombre d'occurrences}`.

```python
>>> count_chars("hello")
{'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

---

### ex02 - Comprehensions

| | |
|---|---|
| Fichier | `comprehensions.py` |
| Interdit | les instructions `for` et `while` (les comprehensions sont autorisees) |

Chaque fonction doit tenir en **une seule ligne** (`return ...`).

- `squares(n)` : la liste des carres de `0` a `n-1`.
- `evens(lst)` : les elements pairs de `lst`, dans l'ordre.
- `flatten(lst)` : aplatit une liste de listes. `[[1, 2], [3], []]` donne `[1, 2, 3]`.
- `word_lengths(sentence)` : un dict `{mot: longueur}`. Indice : `str.split()`.

---

### ex03 - Matrices a la main

| | |
|---|---|
| Fichier | `matrix.py` |
| Interdit | `import numpy` |

Une matrice est representee par une liste de lignes : `[[1, 2, 3], [4, 5, 6]]` est une matrice 2*3 (2 lignes, 3 colonnes).

**Pourquoi cet exercice ?** Une IA, c'est a 90 % des produits de matrices. Tu vas en faire un a la main **une fois**, pour savoir ce que fait NumPy a ta place ensuite.

- `shape(A)` : renvoie le tuple `(lignes, colonnes)`.
- `mat_add(A, B)` : la somme element par element. `ValueError` si les formes different.
- `transpose(A)` : les lignes deviennent les colonnes. Une matrice `(n, m)` devient `(m, n)`.
- `dot(u, v)` : le **produit scalaire** de deux vecteurs, `u[0]*v[0] + u[1]*v[1] + ...`. `ValueError` si les tailles different.
- `mat_mul(A, B)` : le **produit matriciel**. `ValueError` si c'est impossible.

**Le produit matriciel**, la formule a connaitre par coeur : si `A` est `(n, k)` et `B` est `(k, m)`, alors `C = A * B` est `(n, m)`, et

```
C[i][j] = dot(ligne i de A, colonne j de B)
```

Le nombre de **colonnes** de A doit etre egal au nombre de **lignes** de B.

```
[1 2 3]   [ 7  8]   [1*7 + 2*9 + 3*11   1*8 + 2*10 + 3*12]   [ 58  64]
[4 5 6] * [ 9 10] = [4*7 + 5*9 + 6*11   4*8 + 5*10 + 6*12] = [139 154]
          [11 12]
  (2,3)    (3,2)                                               (2,2)
```

Tes fonctions ne doivent **jamais modifier** leurs arguments.

---

### ex04 - La classe Vector

| | |
|---|---|
| Fichier | `vector.py` |
| Interdit | `import numpy` |

Cree une classe `Vector` construite a partir d'une liste de nombres, stockes en `float` :

| Operation | Methode a ecrire | Resultat |
|---|---|---|
| `len(v)` | `__len__` | le nombre d'elements |
| `v[i]` | `__getitem__` | l'element `i` |
| `u + v`, `u - v` | `__add__`, `__sub__` | un **nouveau** Vector. `ValueError` si les tailles different |
| `v * 3` et `3 * v` | `__mul__`, `__rmul__` | multiplication par un scalaire |
| `u == v` | `__eq__` | `True` si memes valeurs |
| `u.dot(v)` | | le produit scalaire |
| `v.norm()` | | la norme : `sqrt(v.dot(v))` |
| `repr(v)` | `__repr__` | `Vector([1.0, 2.0])` |

**`__rmul__` ?** Pour `v * 3`, Python appelle `v.__mul__(3)`. Mais pour `3 * v`, il essaie d'abord `(3).__mul__(v)`, qui echoue (un int ne sait pas multiplier un Vector), puis tente `v.__rmul__(3)`, le " r " voulant dire *right*.

---

### ex05 - Statistiques

| | |
|---|---|
| Fichier | `stats.py` |
| Interdit | `numpy`, `statistics`, `pandas` |

- `mean(lst)` : la moyenne. `ValueError` si la liste est vide.
- `variance(lst)` : la variance **de population**, `moyenne de (x - mean)^2`.
- `std(lst)` : l'ecart-type, `sqrt(variance)`.
- `standardize(lst)` : renvoie la liste des `(x - mean) / std`.
- `load_column(path, col)` : lit un CSV (premiere ligne = noms des colonnes, separateur `,`) et renvoie la colonne `col` sous forme de liste de `float`. `KeyError` si la colonne n'existe pas.

**Pourquoi standardiser ?** Si tu donnes a un modele une surface en m^2 (= 100) et un nombre de pieces (= 3), la surface va ecraser le reste juste parce que ses valeurs sont plus grandes. Standardiser, c'est mettre toutes les variables a la meme echelle : moyenne 0, ecart-type 1. Tu le feras dans **tous** tes futurs projets.

Teste ta fonction sur le vrai fichier `data/appartements.csv` :

```python
print(mean(load_column("../../../data/appartements.csv", "prix")))
```

---

Quand tout est OK :

```bash
python3 moulinette/moulinette.py module00
```

Tu peux passer au [module 01](module01.md).
