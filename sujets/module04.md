# Module 04 — Classification

> *« Oui ou non ? Chat ou chien ? 0, 1, ..., ou 9 ? »*

**Objectif :** prédire une **catégorie** plutôt qu'un nombre. D'abord oui/non (régression logistique), puis N classes (softmax).
**Durée estimée :** 2 à 3 jours.
**Rendu :** `rendu/module04/exXX/`

---

## Le cours

### Le problème

`data/etudiants.csv` contient 200 étudiants :

```
heures_revision,heures_sommeil,reussite
12.1,4.9,1
...
```

On veut prédire `reussite` (1 = admis, 0 = recalé) à partir des heures de révision et de sommeil.

**Pourquoi pas une régression linéaire ?** Elle prédirait des valeurs comme `1.7` ou `-0.3`, qui ne veulent rien dire. On veut une **probabilité**, entre 0 et 1.

### La régression logistique

L'idée : on calcule un score linéaire comme avant, puis on l'écrase entre 0 et 1 avec la **sigmoïde** (module 02) :

```
z = θ₀ + θ₁x₁ + θ₂x₂          ← le "logit", entre -∞ et +∞
p = σ(z) = 1 / (1 + e⁻ᶻ)      ← la probabilité de réussite, entre 0 et 1
```

Et pour décider : si `p ≥ 0.5`, on prédit 1, sinon 0.

**Géométriquement :** `p = 0.5` correspond à `z = 0`, c'est-à-dire `θ₀ + θ₁x₁ + θ₂x₂ = 0`. C'est l'équation d'une **droite**, la **frontière de décision**. D'un côté on prédit 1, de l'autre 0. Le modèle apprend où placer cette droite.

### Pourquoi pas la MSE ?

On pourrait, mais avec une sigmoïde dedans, la MSE devient une surface pleine de plateaux où le gradient est quasi nul : l'apprentissage s'enlise. On utilise une loss faite pour les probabilités.

### La Binary Cross-Entropy

```
BCE = -(1/m) Σᵢ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
```

Ça a l'air barbare, mais regarde les deux cas :

- Si `y = 1`, il reste `-log(p)`. Si `p → 1` (bonne réponse confiante), la loss tend vers 0. Si `p → 0` (mauvaise réponse confiante), la loss tend vers **+∞**.
- Si `y = 0`, il reste `-log(1 - p)`, c'est le symétrique.

**La BCE punit très fort les erreurs commises avec assurance.** Un modèle qui dit « 99 % sûr » et qui se trompe prend cher.

**Piège numérique :** `log(0) = -∞`. Si `p` vaut exactement 0 ou 1, tout explose. Solution : **clipper** `p` dans `[ε, 1 - ε]` avec `ε = 1e-15` (`np.clip`).

### Le gradient (magie des maths)

On applique la règle de la chaîne à `BCE(σ(X_b @ θ))`. Après simplification, en utilisant `σ' = σ(1 - σ)`, presque tout s'annule et il reste :

```
∇BCE(θ) = (1/m) × X_bᵀ @ (p - y)
```

C'est **la même forme** que pour la régression linéaire : erreur × entrée. Ce n'est pas un hasard, et c'est pour ça qu'on associe toujours sigmoïde et cross-entropy.

**Exercice de papier (fortement conseillé) :** démontre-le pour un seul exemple. Pars de `L = -[y log σ(z) + (1-y) log(1-σ(z))]` et montre que `∂L/∂z = σ(z) - y`.

### La sigmoïde stable

`np.exp(-z)` avec `z = -1000` donne `e¹⁰⁰⁰` : overflow. L'astuce :

- Si `z ≥ 0` : `σ(z) = 1 / (1 + e⁻ᶻ)`. Ici `e⁻ᶻ ≤ 1`, donc pas de problème.
- Si `z < 0` : `σ(z) = eᶻ / (1 + eᶻ)`. Même formule (multiplie en haut et en bas par `eᶻ`), mais ici `eᶻ < 1`.

Dans les deux cas, on ne calcule que `e^(-|z|)`, qui est toujours ≤ 1. Calcule `e = np.exp(-np.abs(z))` puis utilise `np.where(condition, si_vrai, si_faux)`.

### Plus de 2 classes : softmax

Pour reconnaître un chiffre de 0 à 9, il faut **10** probabilités qui somment à 1. Le modèle sort 10 scores (logits) `z₀, ..., z₉`, et on applique la **softmax** :

```
softmax(z)ₖ = eᶻᵏ / Σⱼ eᶻʲ
```

- Tous les résultats sont positifs (c'est une exponentielle).
- Ils somment à 1 (on divise par le total).
- Le plus grand score obtient la plus grande probabilité, et l'exponentielle **amplifie** les écarts.

**Exemple :** `z = [2, 1, 0.1]` donne `softmax(z) = [0.66, 0.24, 0.10]`.

**Stabilité :** `softmax(z) = softmax(z - c)` pour n'importe quelle constante `c` (les `e⁻ᶜ` s'annulent en haut et en bas). On soustrait donc le **max de chaque ligne** avant l'exponentielle, et le plus grand exposant devient 0. Plus d'overflow.

**La loss : Cross-Entropy.** Avec `y` l'indice de la bonne classe :

```
CE = -(1/m) Σᵢ log(Pᵢ, yᵢ)
```

On prend juste la probabilité donnée à la **bonne** classe et on en prend `-log`. En NumPy : `P[np.arange(m), y]` sélectionne pour chaque ligne `i` la colonne `y[i]`.

**Le gradient** par rapport aux logits `Z` a lui aussi une forme magique :

```
∂CE/∂Z = (1/m) × (P - Y_onehot)
```

Probabilités prédites moins la vérité. Encore « prédiction - réalité » !

---

## Exercices

> Boucles interdites, sauf la boucle d'entraînement de l'ex03.

### ex00 — Sigmoïde stable

| | |
|---|---|
| Fichier | `sigmoid.py` |

`sigmoid(z)` : marche sur les tableaux, et ne fait **aucun overflow**, même pour `z = ±1000`. La moulinette active `np.errstate(over="raise")` : le moindre overflow et c'est KO.

---

### ex01 — Prédire

| | |
|---|---|
| Fichier | `logistic.py` |

- `predict_proba(X, theta)` : la probabilité `σ(θ₀ + X @ θ[1:])`, de forme `(m,)`.
- `predict(X, theta, threshold=0.5)` : les classes prédites (`1` si proba ≥ seuil, `0` sinon), sous forme d'**entiers** (`.astype(int)`).

---

### ex02 — Binary Cross-Entropy

| | |
|---|---|
| Fichier | `cross_entropy.py` |

- `bce(p, y, eps=1e-15)` : la binary cross-entropy, avec clipping.
- `bce_gradient(X, y, theta)` : le gradient de la BCE par rapport à `theta` (forme `(n+1,)`).

---

### ex03 — Entraîner

| | |
|---|---|
| Fichier | `train.py` |

`fit(X, y, lr=0.1, n_iter=1000)` : comme au module 03, mais avec la sigmoïde et la BCE. `theta` est initialisé à zéro. Renvoie `(theta, history)`.

Dans le `__main__` :

1. Charge `data/etudiants.csv`, standardise, et entraîne.
2. Affiche l'accuracy. Tu devrais être autour de 90 %.
3. **Trace la frontière de décision :** les étudiants en nuage de points (couleur = réussite), avec ta droite `θ₀ + θ₁x₁ + θ₂x₂ = 0` par-dessus. C'est très satisfaisant.
4. Combien de chances de réussite pour quelqu'un qui révise 8 h et dort 7 h ? N'oublie pas de standardiser cette entrée avec les mêmes paramètres.

---

### ex04 — Softmax

| | |
|---|---|
| Fichier | `softmax.py` |

- `softmax(Z)` : `Z` est `(m, K)`. Applique la softmax sur **chaque ligne**, de façon stable. Pense à `keepdims=True`.
- `cross_entropy(P, y)` : `P` est `(m, K)` (des probabilités), `y` est `(m,)` (des indices de classe). Renvoie la cross-entropy moyenne.
- `softmax_ce_gradient(Z, y)` : le gradient de `cross_entropy(softmax(Z), y)` par rapport à `Z`, de forme `(m, K)`.

Ces trois fonctions sont le cœur de la sortie de **tout** classifieur moderne, de MNIST à GPT. Tu les réutiliseras au module 06.

---

```bash
python3 moulinette/moulinette.py module04
```

Jusqu'ici, tu calcules les gradients à la main. Ça devient vite infernal avec plusieurs couches… Au [module 05](module05.md), tu vas construire une machine qui les calcule **toute seule**.
