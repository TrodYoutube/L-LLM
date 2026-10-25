# Module 05 — Autograd : le moteur de la backpropagation

> *« Je n'ai jamais calculé un gradient de ma vie depuis que j'ai écrit cette classe. »*

**Objectif :** coder une classe `Value` qui retient **chaque opération** effectuée sur elle, et qui calcule **automatiquement** tous les gradients via la règle de la chaîne. C'est exactement ce que fait PyTorch, en version miniature (environ 100 lignes).
**Durée estimée :** 3 à 5 jours. C'est le module le plus formateur de la piscine.
**Rendu :** `rendu/module05/exXX/`

---

## Le cours

### Le problème

Au module 04, tu as calculé `∇BCE` à la main. Pour un réseau à 3 couches, le calcul à la main devient un cauchemar, et pour GPT (des centaines de couches), c'est impossible. Il faut que l'ordinateur le fasse.

### L'idée : le graphe de calcul

Toute expression mathématique est un **graphe** d'opérations élémentaires. Prends `L = (a × b) + c` avec `a = 2`, `b = -3` et `c = 10` :

```
  a=2 ──┐
        (×)── e=-6 ──┐
 b=-3 ──┘            (+)── L=4
         c=10 ───────┘
```

Chaque nœud connaît :
- sa **valeur** (`data`),
- ses **parents** (les nœuds qui ont servi à le calculer),
- l'**opération** qui l'a créé.

C'est une structure chaînée, comme tes listes chaînées en C, sauf que chaque nœud peut avoir plusieurs parents. C'est un **DAG** (graphe orienté acyclique).

### Le backward : la règle de la chaîne, nœud par nœud

On veut `∂L/∂a`, `∂L/∂b` et `∂L/∂c`. On remonte le graphe **depuis la fin** :

1. `∂L/∂L = 1`, évidemment.
2. **Nœud `+`** (`L = e + c`) : `∂L/∂e = 1` et `∂L/∂c = 1`. Une addition **transmet le gradient tel quel** à ses deux parents.
3. **Nœud `×`** (`e = a × b`) : `∂e/∂a = b` et `∂e/∂b = a`. Règle de la chaîne :
   - `∂L/∂a = ∂L/∂e × ∂e/∂a = 1 × b = -3`
   - `∂L/∂b = ∂L/∂e × ∂e/∂b = 1 × a = 2`

Le mécanisme général : **chaque nœud reçoit le gradient de sa sortie (`out.grad`), le multiplie par sa dérivée locale, et l'envoie à ses parents.**

| Opération | `out = ...` | Gradient envoyé aux parents |
|---|---|---|
| addition | `a + b` | `a.grad += out.grad` et `b.grad += out.grad` |
| multiplication | `a × b` | `a.grad += b.data × out.grad` et `b.grad += a.data × out.grad` |
| puissance | `aⁿ` | `a.grad += n × a.dataⁿ⁻¹ × out.grad` |
| exp | `eᵃ` | `a.grad += out.data × out.grad` |
| tanh | `tanh(a)` | `a.grad += (1 - out.data²) × out.grad` |
| relu | `max(0, a)` | `a.grad += (1 si a.data > 0 sinon 0) × out.grad` |

### Pourquoi `+=` et pas `=` ?

Si une variable est utilisée **plusieurs fois**, par exemple `y = x × x + x`, elle reçoit du gradient par **plusieurs chemins**, et il faut les **additionner** (règle de la chaîne multivariée). Avec `=`, le dernier chemin écraserait les autres. C'est LE bug classique, et la moulinette le teste.

```
y = x*x + x    →    dy/dx = 2x + 1 = 7 pour x = 3
```

### L'ordre : le tri topologique

Pour qu'un nœud envoie son gradient, il faut qu'il ait **fini de recevoir** le sien, donc que tous ses enfants aient déjà été traités. On construit d'abord une liste ordonnée des nœuds (**tri topologique**) via un parcours en profondeur (DFS) :

```python
topo = []
visited = set()
def build(v):
    if v not in visited:
        visited.add(v)
        for parent in v._prev:
            build(parent)
        topo.append(v)      # un nœud est ajouté APRÈS tous ses parents
build(self)
```

Puis on met `self.grad = 1` et on appelle `_backward()` de chaque nœud dans l'ordre **inverse** de `topo`.

### Comment stocker le `_backward` ? Les closures

Chaque nœud doit savoir comment propager son gradient. On stocke une **fonction** dans le nœud, définie au moment où on crée l'opération :

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

La fonction interne `_backward` « capture » les variables `self`, `other` et `out` : c'est une **closure**. En C, ça reviendrait à stocker un pointeur sur fonction **plus** une struct contenant les pointeurs vers `self`, `other` et `out`. Python fait ça tout seul.

Par défaut (pour les feuilles comme `a`, `b`, `c`), `_backward` ne fait rien : `self._backward = lambda: None`.

### Économiser du code

Une fois que tu as `+`, `*` et `**`, le reste se déduit :

- `-a` revient à `a * -1`
- `a - b` revient à `a + (-b)`
- `a / b` revient à `a * b**-1`

Les gradients suivent tout seuls, puisque ce sont des compositions d'opérations que tu sais déjà dériver !

### Du neurone au réseau

Avec `Value`, un neurone à `n` entrées devient :

```
out = tanh(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

où chaque `wᵢ` et `b` est une `Value`. Une **couche** (`Layer`) est une liste de neurones qui reçoivent **les mêmes entrées**. Un **MLP** (*Multi-Layer Perceptron*) est une suite de couches, où la sortie de l'une est l'entrée de la suivante :

```
entrées (3)  →  Layer(3→4)  →  Layer(4→4)  →  Layer(4→1)  →  sortie
               4 neurones      4 neurones      1 neurone
```

Pour l'entraîner : forward, loss, `zero_grad()` (remettre tous les grads à 0, sinon ils s'accumulent d'une itération à l'autre !), `loss.backward()`, puis `p.data -= lr * p.grad` pour chaque paramètre.

---

## Exercices

Les exercices construisent **la même classe** petit à petit. Pour chaque exercice, copie ton `value.py` précédent dans le nouveau dossier et complète-le.

### ex00 — Le graphe

| | |
|---|---|
| Fichier | `value.py` |

Classe `Value(data, _children=(), _op="")` avec :

- les attributs `data`, `_prev` (un `set` des parents) et `_op` (une str : `"+"`, `"*"`…) ;
- les opérations `+`, `*`, `-` (binaire et unaire) qui renvoient de **nouvelles** `Value`, et qui acceptent aussi des nombres : `a + 1`, `2 * a`, `1 - a`. Pense à `__radd__`, `__rmul__` et `__rsub__` ;
- `__repr__` qui renvoie `Value(data=1.5)`.

Le nœud créé par `a * b` doit avoir `_op == "*"` et `_prev == {a, b}`.

**Attention :** `a - b` implémenté comme `a + (-b)` donne un nœud final `+`, et c'est très bien comme ça.

---

### ex01 — Le backward

| | |
|---|---|
| Fichier | `value.py` |

Ajoute :

- l'attribut `grad`, initialisé à `0.0` ;
- un `_backward` pour chaque opération (closures) ;
- la méthode `backward()` : tri topologique, `self.grad = 1.0`, puis appel des `_backward` dans l'ordre inverse.

Teste à la main avant la moulinette :

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = a * b + c
L.backward()
print(a.grad, b.grad, c.grad)   # -3.0 2.0 1.0
```

---

### ex02 — Plus d'opérations

| | |
|---|---|
| Fichier | `value.py` |

Ajoute :

- `__pow__(k)` où `k` est un int ou un float (pas une `Value`) ;
- `__truediv__` et `__rtruediv__` (`a / b` et `2 / a`), via `**-1` ;
- les méthodes `exp()`, `tanh()` et `relu()`.

La moulinette compare tes gradients avec des dérivées numériques sur une expression bien tordue.

---

### ex03 — Un réseau de neurones

| | |
|---|---|
| Fichiers | `value.py` (copie de l'ex02) et `nn.py` |

Dans `nn.py` (qui commence par `from value import Value`) :

- `Neuron(n_in, nonlin=True)` : `n_in` poids initialisés avec `random.uniform(-1, 1)` et un biais à `0`. `neuron(x)`, avec `x` une liste de nombres ou de `Value`, renvoie `tanh(Σ wᵢxᵢ + b)` si `nonlin`, sinon la somme brute. Méthode `parameters()` : la liste des poids puis du biais.
- `Layer(n_in, n_out, nonlin=True)` : `n_out` neurones. `layer(x)` renvoie la **liste** des sorties.
- `MLP(n_in, sizes)` : par exemple `MLP(3, [4, 4, 1])`. La **dernière** couche est linéaire (`nonlin=False`). `mlp(x)` renvoie une `Value` seule si la dernière couche a un neurone, une liste sinon.
- `parameters()` et `zero_grad()` pour `Layer` et `MLP`. Astuce : une classe mère `Module` qui implémente `zero_grad()` en s'appuyant sur `parameters()`.

Pour faire `__call__` : `def __call__(self, x):` rend un objet appelable comme une fonction.

Dans le `__main__`, entraîne ton MLP sur ce mini-dataset et affiche la loss qui descend :

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
```

**Félicitations** : à ce stade, tu as codé de zéro un framework de deep learning. PyTorch fait **exactement** ça, sauf qu'au lieu de manipuler des nombres un par un, il manipule des **tenseurs** (des tableaux NumPy), ce qui le rend des millions de fois plus rapide.

---

```bash
python3 moulinette/moulinette.py module05
```

Ton `Value` est génial pour comprendre, mais trop lent pour de vraies données. Au [module 06](module06.md), on applique les mêmes idées à des **matrices** NumPy, et on attaque MNIST.
