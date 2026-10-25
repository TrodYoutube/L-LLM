# Module 02 — Les maths de l'IA

> *« Apprendre, c'est descendre une colline les yeux bandés, en tâtant la pente avec ses pieds. »*

**Objectif :** comprendre les **dérivées**, le **gradient** et la **descente de gradient**. C'est le moteur de TOUTE l'IA moderne.
**Durée estimée :** 2 à 4 jours. Prends ton temps, c'est le module le plus important.
**Rendu :** `rendu/module02/exXX/`

---

## Le cours

### 1. La dérivée : « si je bouge un peu x, de combien bouge f(x) ? »

Prends `f(x) = x²`. En `x = 3`, `f(3) = 9`. Si je bouge `x` d'un tout petit peu, disons `h = 0.001` :

```
f(3.001) = 9.006001
```

`f` a bougé de `0.006`, soit **6 fois** plus que `x`. La dérivée de `f` en 3 vaut 6, et on note `f'(3) = 6`.

**Formellement :**

```
f'(x) = limite quand h → 0 de  (f(x + h) - f(x)) / h
```

**Ce que ça veut dire concrètement :**

- `f'(x) > 0` : f **monte** quand x augmente.
- `f'(x) < 0` : f **descend** quand x augmente.
- `f'(x) = 0` : on est sur un plat, peut-être un **minimum**.
- `|f'(x)|` grand : la pente est raide.

### 2. Les dérivées à connaître

| f(x) | f'(x) |
|---|---|
| `c` (constante) | `0` |
| `x` | `1` |
| `a * x` | `a` |
| `x²` | `2x` |
| `xⁿ` | `n * xⁿ⁻¹` |
| `eˣ` (`exp`) | `eˣ` (elle est sa propre dérivée !) |
| `ln(x)` (`log`) | `1 / x` |
| `u + v` | `u' + v'` |
| `u * v` | `u' * v + u * v'` |

### 3. La règle de la chaîne (THE règle)

Quand on **compose** des fonctions, les dérivées se **multiplient** :

```
si  f(x) = g(h(x))   alors   f'(x) = g'(h(x)) × h'(x)
```

**Exemple :** `f(x) = (3x + 1)²`. On pose `h(x) = 3x + 1` et `g(u) = u²`.

- `h'(x) = 3`
- `g'(u) = 2u`, donc `g'(h(x)) = 2(3x + 1)`
- `f'(x) = 2(3x + 1) × 3 = 6(3x + 1)`

**Intuition :** si `h` va 3 fois plus vite que `x`, et `g` va `2u` fois plus vite que `h`, alors `g` va `3 × 2u` fois plus vite que `x`. Comme des engrenages.

**Pourquoi c'est THE règle :** un réseau de neurones est une **énorme composition de fonctions** : `loss(couche3(couche2(couche1(x))))`. Pour savoir comment chaque poids influence la loss, on applique la règle de la chaîne de la fin vers le début. Ça s'appelle la **backpropagation**, et c'est juste ça.

### 4. La dérivée numérique

Pas envie de calculer la dérivée à la main ? On peut l'**approximer** avec un petit `h` :

```
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)        ← différence CENTRÉE
```

Pourquoi `x + h` et `x - h`, plutôt que `x + h` et `x` ? Parce que l'erreur est beaucoup plus faible (en `h²` au lieu de `h`). Fais le test avec `exp` : tu verras.

**Alors pourquoi ne pas toujours faire ça ?** Parce qu'un réseau a des **millions** de paramètres, et il faudrait 2 évaluations du réseau entier **par paramètre**. La règle de la chaîne calcule tous les gradients en **un seul passage**. Mais la dérivée numérique reste précieuse pour **vérifier** que tes gradients sont justes : c'est ce que fait la moulinette (« gradient check »).

### 5. Plusieurs variables : le gradient

Maintenant `f` dépend de plusieurs variables : `f(x, y) = x² + 3y²`.

La **dérivée partielle** par rapport à `x`, notée `∂f/∂x`, c'est la dérivée en faisant comme si `y` était une constante :

```
∂f/∂x = 2x          ∂f/∂y = 6y
```

Le **gradient**, noté `∇f`, c'est le vecteur de toutes les dérivées partielles :

```
∇f(x, y) = [2x, 6y]
```

**Propriété fondamentale :** le gradient pointe dans la direction où `f` **augmente le plus vite**. Donc `-∇f` pointe vers la descente la plus raide.

### 6. La descente de gradient

Imagine que tu es sur une montagne dans le brouillard et que tu veux atteindre la vallée. Tu ne vois rien, mais tu sens la pente sous tes pieds. Stratégie : faire un petit pas dans la direction qui descend, puis recommencer.

```
répéter :
    x = x - lr * ∇f(x)
```

`lr` (*learning rate*) est la taille du pas.

- **Trop petit :** tu descends très lentement.
- **Trop grand :** tu sautes par-dessus la vallée, et tu peux même diverger (f explose).
- **Bien choisi :** tu converges vers le minimum.

**Exemple à la main :** `f(x) = (x - 3)²`, donc `f'(x) = 2(x - 3)`. On part de `x = 0` avec `lr = 0.1` :

```
x = 0     → f'(0) = -6     → x = 0 - 0.1 × (-6) = 0.6
x = 0.6   → f'(0.6) = -4.8 → x = 0.6 + 0.48 = 1.08
x = 1.08  → ...            → x = 1.464
...                         → x → 3   ✓
```

**Le lien avec l'IA :** dans un modèle, `x`, ce sont les **paramètres** (les poids), et `f`, c'est la **loss** (l'erreur). Entraîner un modèle, c'est faire une descente de gradient sur la loss. **C'est tout.**

### 7. Les fonctions d'activation

Deux fonctions qu'on retrouve partout dans les réseaux de neurones :

**Sigmoïde :** écrase n'importe quel nombre entre 0 et 1, parfait pour une probabilité.

```
σ(x) = 1 / (1 + e⁻ˣ)

σ(-∞) = 0     σ(0) = 0.5     σ(+∞) = 1
```

Sa dérivée a une forme magique : `σ'(x) = σ(x) × (1 - σ(x))`. Tu peux la redémontrer avec la règle de la chaîne, c'est un bon exercice.

**ReLU** (*Rectified Linear Unit*) : la plus utilisée aujourd'hui, car bête et efficace.

```
relu(x) = max(0, x)        relu'(x) = 1 si x > 0, sinon 0
```

### 8. Ton premier neurone

Un neurone avec une entrée, c'est :

```
neuron(x, w, b) = σ(w × x + b)
```

Pour l'entraîner, il faut savoir comment la sortie change quand on bouge `w` et `b`. Avec la règle de la chaîne, en posant `z = w × x + b` :

```
∂neuron/∂w = σ'(z) × ∂z/∂w = σ'(z) × x
∂neuron/∂b = σ'(z) × ∂z/∂b = σ'(z) × 1
```

Tu viens de faire de la backprop sur un neurone. 🎉

---

## Exercices

### ex00 — Dérivée numérique

| | |
|---|---|
| Fichier | `derivative.py` |

`derivative(f, x, h=1e-5)` : renvoie une approximation de `f'(x)` par **différence centrée**.

```python
>>> derivative(lambda x: x ** 2, 3.0)
6.000000000039306
```

`lambda x: x ** 2` est une fonction anonyme en une ligne, l'équivalent d'un pointeur sur fonction en C.

---

### ex01 — Gradient numérique

| | |
|---|---|
| Fichier | `gradient.py` |

`gradient(f, x, h=1e-5)` : `x` est un `np.ndarray` 1D et `f` prend un vecteur et renvoie un nombre. Renvoie le gradient `∇f(x)`, un tableau de même forme que `x`.

Pour chaque composante `i`, calcule la dérivée en ne bougeant **que** `x[i]`. Ne modifie pas le `x` reçu en argument (pense à `.copy()`). Ici, une boucle sur les composantes est autorisée.

---

### ex02 — Descente de gradient

| | |
|---|---|
| Fichier | `descent.py` |

`gradient_descent(grad_f, x0, lr, n_iter)` :

- `grad_f` : une fonction qui renvoie le gradient en un point.
- `x0` : le point de départ (un float ou un `np.ndarray`).
- Fait `n_iter` pas de `x = x - lr * grad_f(x)`.
- Renvoie le tuple `(x_final, history)`, où `history` est la liste des positions : `[x0, x1, ..., x_n_iter]`, soit `n_iter + 1` éléments.

**Bonus (non noté, mais fais-le) :** trace l'évolution de `f(x)` avec matplotlib pour plusieurs learning rates (`0.01`, `0.1`, `0.9`, `1.1`). Observe la convergence lente, rapide, l'oscillation et la divergence.

```python
import matplotlib.pyplot as plt
plt.plot([f(x) for x in history], label="lr=0.1")
plt.legend()
plt.show()
```

---

### ex03 — La règle de la chaîne

| | |
|---|---|
| Fichier | `chain_rule.py` |

Toutes ces fonctions doivent marcher sur des nombres **et** sur des `np.ndarray` (utilise `np.exp`, `np.maximum`…) :

- `sigmoid(x)` et `sigmoid_prime(x)`.
- `relu(x)` et `relu_prime(x)`, avec `relu_prime(x) = 0` pour `x <= 0`.
- `f(x) = sigmoid(3x² + 1)` et `f_prime(x)`, sa dérivée **calculée à la main** avec la règle de la chaîne. Pas de dérivée numérique ici !
- `neuron(x, w, b) = sigmoid(w * x + b)` et `neuron_grad(x, w, b)`, qui renvoie le tuple `(∂neuron/∂w, ∂neuron/∂b)`.

La moulinette compare tes dérivées analytiques avec des dérivées numériques. Si elles ne correspondent pas, c'est ta règle de la chaîne qui est fausse.

---

```bash
python3 moulinette/moulinette.py module02
```

Tu as maintenant tous les outils. Au [module 03](module03.md), ton premier modèle va **apprendre**.
