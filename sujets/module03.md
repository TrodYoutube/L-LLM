# Module 03 — Régression linéaire

> *« Tout modèle est faux, mais certains sont utiles. » (George Box)*

**Objectif :** construire ton premier modèle qui **apprend à partir de données** : prédire le prix d'un appartement à partir de sa surface.
**Durée estimée :** 2 à 3 jours.
**Rendu :** `rendu/module03/exXX/`

---

## Le cours

### Le problème

Tu as un fichier `data/appartements.csv` avec 120 appartements :

```
surface,prix
119,496.5
74,293.0
...
```

(Les prix sont en k€.) Question : combien vaut un appartement de 50 m² ?

Trace les données (`plt.scatter(surface, prix)`) : les points forment à peu près une **droite**. L'idée, c'est de trouver **la droite qui passe au mieux** à travers ces points.

### Le modèle

Une droite, c'est :

```
prix_prédit = θ₀ + θ₁ × surface
```

- `θ₁` (thêta 1) : la pente, c'est-à-dire combien de k€ coûte chaque m² supplémentaire.
- `θ₀` : l'ordonnée à l'origine, aussi appelée **biais**.

`θ = [θ₀, θ₁]` sont les **paramètres** du modèle. Au début on ne les connaît pas : c'est le modèle qui va les **apprendre**.

**Avec plusieurs variables** (surface, nombre de pièces, étage…), c'est pareil :

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

En notation matricielle, pour `m` exemples d'un coup, avec `X` de forme `(m, n)` et `θ` de forme `(n + 1,)` :

```
ŷ = θ₀ + X @ θ[1:]          ← shape (m,)
```

**L'astuce du biais :** si on ajoute une colonne de 1 au début de `X` (ton `add_bias` du module 01 !), alors `ŷ = X_b @ θ`. Plus besoin de traiter `θ₀` à part.

```
X_b = [[1, x₁⁽¹⁾, x₂⁽¹⁾],        θ = [θ₀,         X_b @ θ = [θ₀ + θ₁x₁⁽¹⁾ + θ₂x₂⁽¹⁾,
       [1, x₁⁽²⁾, x₂⁽²⁾],             θ₁,                    θ₀ + θ₁x₁⁽²⁾ + θ₂x₂⁽²⁾,
       ...]                           θ₂]                    ...]
```

### La loss : mesurer l'erreur

Comment dire si une droite est « bonne » ? On mesure l'écart entre prédiction et réalité, pour chaque point, et on fait la moyenne des carrés. C'est la **MSE** (*Mean Squared Error*) :

```
J(θ) = (1/m) × Σᵢ (ŷᵢ - yᵢ)²
```

Pourquoi **au carré** ?
1. Pour que les erreurs positives et négatives ne s'annulent pas.
2. Pour punir beaucoup plus les grosses erreurs.
3. Parce que c'est dérivable partout (contrairement à la valeur absolue en 0).

**Le but de l'apprentissage : trouver le θ qui minimise J(θ).**

Et comment on minimise une fonction ? **Descente de gradient** (module 02) !

### Le gradient de la MSE

Il faut `∂J/∂θⱼ` pour chaque paramètre. Règle de la chaîne :

```
J = (1/m) Σᵢ (ŷᵢ - yᵢ)²       avec  ŷᵢ = Σⱼ θⱼ X_b[i, j]

∂J/∂θⱼ = (1/m) Σᵢ 2(ŷᵢ - yᵢ) × ∂ŷᵢ/∂θⱼ
       = (2/m) Σᵢ (ŷᵢ - yᵢ) × X_b[i, j]
```

Et en version vectorisée, pour tous les `j` d'un coup :

```
∇J(θ) = (2/m) × X_bᵀ @ (ŷ - y)            ← shape (n + 1,)
```

Vérifie les shapes : `X_bᵀ` est `(n+1, m)` et `(ŷ - y)` est `(m,)`, donc le résultat est `(n+1,)`. ✓

**Intuition :** `(ŷ - y)` est le vecteur des erreurs. Chaque paramètre `θⱼ` est corrigé proportionnellement à l'erreur **multipliée par la variable qu'il contrôle**. Si une variable n'a rien à voir avec l'erreur, son paramètre bouge peu.

### L'algorithme d'entraînement

```
θ = [0, 0, ..., 0]
répéter n_iter fois :
    ŷ = X_b @ θ                   # 1. forward : prédire
    loss = mse(ŷ, y)              # 2. mesurer l'erreur
    grad = (2/m) X_bᵀ (ŷ - y)     # 3. calculer le gradient
    θ = θ - lr × grad             # 4. mettre à jour
```

Ces 4 étapes, **forward, loss, gradient, update**, tu vas les retrouver dans TOUS les modèles de cette piscine, jusqu'au GPT.

### Le piège : l'échelle des variables

Si tu entraînes directement sur les surfaces (15 à 150), les gradients sont énormes, et il faut un `lr` minuscule sinon ça diverge. **Solution : standardiser `X`** (module 01). Une fois entraîné sur des données standardisées, pour prédire une nouvelle valeur, il faut lui appliquer **la même transformation**, avec la moyenne et l'écart-type du jeu d'entraînement.

### Évaluer : le score R²

La MSE dépend de l'unité (k€²), ce qui la rend difficile à interpréter. Le **R²** est plus parlant :

```
R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

- `R² = 1` : prédiction parfaite.
- `R² = 0` : pas mieux que de toujours prédire la moyenne.
- `R² < 0` : pire que la moyenne (bravo…).

### La solution exacte (bonus culture)

Pour la régression linéaire, il existe une formule directe, l'**équation normale** :

```
θ = (X_bᵀ X_b)⁻¹ X_bᵀ y
```

Alors pourquoi la descente de gradient ? Parce que cette formule ne marche **que** pour la régression linéaire, et devient très lente avec beaucoup de variables. La descente de gradient marche pour **tout**, y compris GPT. Mais elle est pratique pour vérifier ton `fit`.

---

## Exercices

> Boucles interdites dans les ex00, ex01, ex02 et ex04. Dans l'ex03, seule la boucle d'entraînement est autorisée.

### ex00 — Prédire

| | |
|---|---|
| Fichier | `predict.py` |

`predict(X, theta)` : `X` est `(m, n)` et `theta` est `(n + 1,)`, avec `theta[0]` comme biais. Renvoie `ŷ` de forme `(m,)`.

---

### ex01 — Mesurer

| | |
|---|---|
| Fichier | `loss.py` |

- `mse(y_pred, y)` : la mean squared error.
- `r2_score(y_pred, y)` : le score R².

---

### ex02 — Le gradient

| | |
|---|---|
| Fichier | `gradient.py` |

`gradient(X, y, theta)` : le gradient de la MSE par rapport à `theta`, de forme `(n + 1,)`. Utilise exactement la formule du cours, avec le facteur `2/m`.

---

### ex03 — Entraîner

| | |
|---|---|
| Fichier | `train.py` |

`fit(X, y, lr=0.01, n_iter=1000)` :

- Initialise `theta` à zéro (forme `(n + 1,)`).
- Fait `n_iter` itérations de descente de gradient.
- Renvoie `(theta, history)`, où `history` est la liste des loss (MSE) **avant** chaque mise à jour, soit `n_iter` éléments.

Ajoute un `if __name__ == "__main__":` qui :

1. Charge `data/appartements.csv` (par exemple avec `np.loadtxt(path, delimiter=",", skiprows=1)`).
2. Standardise la surface, en gardant la moyenne et l'écart-type de côté.
3. Entraîne le modèle.
4. Affiche le R² et le prix prédit pour un appartement de **50 m²**. Tu devrais trouver environ 215 k€.
5. Trace deux graphiques : les points avec ta droite par-dessus, et la courbe de la loss en fonction des itérations.

Pour trouver le CSV depuis ton dossier d'exercice :

```python
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
```

---

### ex04 — L'équation normale

| | |
|---|---|
| Fichier | `normal_equation.py` |

`normal_equation(X, y)` : renvoie le `theta` optimal avec la formule exacte. Utilise `np.linalg.solve(A, b)` plutôt que `np.linalg.inv(A) @ b` : c'est plus stable et plus rapide. Compare avec le résultat de ton `fit`.

---

```bash
python3 moulinette/moulinette.py module03
```

Tu sais prédire un nombre. Et si on voulait prédire une **catégorie** ? [Module 04](module04.md).
