# Module 06 — Un réseau de neurones from scratch (MNIST)

> *« Le "Hello World" du deep learning : reconnaître des chiffres écrits à la main. »*

**Objectif :** coder un vrai réseau de neurones **en NumPy pur**, avec des couches, la backprop matricielle et l'entraînement par mini-batch, puis atteindre **≥ 97 %** de précision sur MNIST.
**Durée estimée :** 4 à 6 jours.
**Rendu :** `rendu/module06/exXX/`
**Prérequis :** `./data/get_data.sh` lancé (les fichiers MNIST sont dans `data/mnist/`).

---

## Le cours

### MNIST

70 000 images de chiffres manuscrits (0 à 9), en 28×28 pixels et niveaux de gris (0 = noir, 255 = blanc). 60 000 servent à l'entraînement et 10 000 au **test**.

**Le test set ne sert JAMAIS à entraîner.** Il sert uniquement à mesurer si ton modèle **généralise** à des images qu'il n'a jamais vues. Un modèle qui a 100 % sur le train et 60 % sur le test a appris par cœur : c'est l'**overfitting**.

Chaque image est « aplatie » en un vecteur de 784 pixels (28 × 28). Un batch de 64 images est donc une matrice `(64, 784)`.

### Le format IDX (ça va te rappeler le C)

Les fichiers MNIST sont en binaire brut, compressés en gzip. Format :

```
Images (train-images-idx3-ubyte) :
[offset] [type]          [description]
0000     int32 big-endian magic number = 2051
0004     int32 big-endian nombre d'images
0008     int32 big-endian nombre de lignes (28)
0012     int32 big-endian nombre de colonnes (28)
0016     uint8           pixel, pixel, pixel, ...  (ligne par ligne)

Labels (train-labels-idx1-ubyte) :
0000     int32 big-endian magic number = 2049
0004     int32 big-endian nombre de labels
0008     uint8           label, label, ...
```

En Python :

```python
import gzip, struct
with gzip.open(path, "rb") as f:          # comme fopen, mais décompresse à la volée
    (magic,) = struct.unpack(">I", f.read(4))   # ">I" = un uint32 big-endian
    data = f.read()                              # tout le reste, en bytes
pixels = np.frombuffer(data, dtype=np.uint8)     # bytes → tableau NumPy, sans copie
```

### Un réseau en matrices

Au module 05, chaque neurone était un objet. Maintenant, une **couche entière** est un produit matriciel :

```
X : (batch, n_in)       W : (n_in, n_out)       b : (n_out,)

Z = X @ W + b           → (batch, n_out)
```

Chaque colonne de `W` contient les poids d'un neurone. Et on traite tout le batch d'un coup.

Notre réseau pour MNIST :

```
X (64, 784) → Dense(784→128) → ReLU → Dense(128→10) → logits (64, 10) → Softmax + CrossEntropy → loss
```

### La backprop matricielle

Chaque couche a deux méthodes :

- **`forward(X)`** : calcule la sortie, et **mémorise** ce dont elle aura besoin pour le backward (ici, `X`).
- **`backward(dout)`** : reçoit `dout = ∂L/∂sortie` (même forme que la sortie), calcule les gradients de **ses paramètres**, et renvoie `∂L/∂entrée` pour la couche précédente.

C'est exactement ton `_backward` du module 05, mais pour des blocs entiers.

**Couche Dense** (`Z = X @ W + b`) :

```
dW = Xᵀ @ dout        (n_in, batch) @ (batch, n_out) = (n_in, n_out)    ✓ même forme que W
db = dout.sum(axis=0)  (n_out,)                                           ✓ même forme que b
dX = dout @ Wᵀ        (batch, n_out) @ (n_out, n_in) = (batch, n_in)     ✓ même forme que X
```

**L'astuce pour ne jamais retenir ces formules :** le gradient d'un tableau a **toujours la même forme** que ce tableau. Il n'y a souvent qu'une seule façon de combiner les matrices pour obtenir la bonne forme. Écris les shapes, et la formule se déduit.

Pourquoi `db` est une **somme** sur le batch ? Parce que `b` est ajouté à **chaque** exemple du batch (broadcasting). Il est donc « utilisé plusieurs fois », et ses gradients s'**additionnent** : c'est ton `+=` du module 05 !

**ReLU** (élément par élément) :

```
forward :  out = X * (X > 0)          → on mémorise le masque (X > 0)
backward : dX = dout * masque          → le gradient passe là où X > 0, sinon il est bloqué
```

**Softmax + Cross-Entropy** : tu l'as fait au module 04 : `dZ = (P - Y_onehot) / batch`.

### L'initialisation des poids

Si tous les poids valent 0, tous les neurones calculent la même chose et reçoivent le même gradient : ils restent identiques pour toujours, et ton réseau n'a qu'un seul neurone utile. Il faut de l'**aléatoire**.

Mais pas n'importe lequel : trop grand, les activations explosent de couche en couche ; trop petit, elles s'éteignent. Pour ReLU, on utilise l'**initialisation de He** :

```
W = rng.standard_normal((n_in, n_out)) * sqrt(2 / n_in)
b = 0
```

### Le mini-batch SGD

Calculer le gradient sur les 60 000 images à chaque pas est lent. Et le calculer sur 1 seule image donne un gradient trop bruité. Le compromis, c'est le **mini-batch** :

```
pour chaque epoch :
    mélanger les données (shuffle)
    pour chaque paquet de 64 images :
        forward → loss → backward → mise à jour des poids
    mesurer l'accuracy sur le test set
```

Une **epoch** correspond à un passage sur toutes les données, soit 60 000 / 64 ≈ 938 mises à jour. Avec un bon réseau, 5 à 10 epochs suffisent pour dépasser 97 %.

### Le gradient check

Comment être sûr que ton backward est juste ? Tu compares avec la **dérivée numérique** (module 02) sur un petit exemple. Si l'erreur relative est inférieure à 1e-5, c'est bon. La moulinette le fait pour toi, mais **apprends à le faire toi-même** : c'est l'outil de debug n°1 quand on code des gradients.

---

## Exercices

### ex00 — Charger MNIST

| | |
|---|---|
| Fichier | `mnist.py` |
| Autorisé | `gzip`, `struct`, `numpy` |

- `load_images(path)` : lit un fichier `.gz` d'images IDX et renvoie un tableau `float32` de forme `(N, lignes × colonnes)`, avec des valeurs entre 0 et 1 (divise par 255).
- `load_labels(path)` : lit un fichier `.gz` de labels et renvoie un tableau d'entiers de forme `(N,)`.
- Les deux lèvent une `ValueError` si le magic number est incorrect. **Lis et vérifie le magic number AVANT de lire le reste de l'en-tête.**

**Bonus :** affiche quelques images avec `plt.imshow(X[i].reshape(28, 28), cmap="gray")` et leurs labels en titre.

---

### ex01 — Les couches

| | |
|---|---|
| Fichier | `layers.py` |

**`Dense(n_in, n_out, rng=None)`**
- Attributs `W` `(n_in, n_out)` avec initialisation de He, et `b` `(n_out,)` à zéro. Si `rng` vaut `None`, utilise `np.random.default_rng()`.
- `forward(X)` : renvoie `X @ W + b` et mémorise `X`.
- `backward(dout)` : stocke `self.dW` et `self.db`, renvoie `dX`.

**`ReLU()`**
- `forward(X)` et `backward(dout)`.

---

### ex02 — La loss

| | |
|---|---|
| Fichier | `loss.py` |

**`SoftmaxCrossEntropy()`**
- `forward(logits, y)` : `logits` est `(batch, K)`, `y` est `(batch,)` (des entiers). Renvoie la cross-entropy moyenne (un float). Doit être stable numériquement.
- `backward()` : renvoie `∂loss/∂logits`, de forme `(batch, K)`.

**Astuce de pro (log-softmax) :** au lieu de calculer `P = softmax(Z)` puis `log(P)`, calcule directement `log P = Z' - log(Σ exp(Z'))` avec `Z' = Z - max(Z)`. Plus de `log(0)` possible.

---

### ex03 — Le réseau

| | |
|---|---|
| Fichiers | `layers.py`, `loss.py` (copies) et `network.py` |

**`Network(layers, loss)`**, où `layers` est une liste de couches et `loss` un `SoftmaxCrossEntropy` :

- `forward(X)` : passe `X` dans toutes les couches, renvoie les logits.
- `train_step(X, y, lr)` : forward, loss, backward (couches **à l'envers**), puis mise à jour SGD de toutes les couches qui ont un `W` (`W -= lr * dW`, idem pour `b`). Renvoie la loss.
- `predict(X)` : renvoie la classe prédite pour chaque ligne.

La moulinette entraîne ton réseau sur une **spirale** à 3 bras : un problème qu'aucun modèle linéaire ne peut résoudre. Il faut plus de 90 % de précision.

```python
net = Network([Dense(2, 64), ReLU(), Dense(64, 3)], SoftmaxCrossEntropy())
```

**Bonus visuel :** trace la frontière de décision de ton réseau sur la spirale (une grille de points avec `np.meshgrid`, puis `plt.contourf`). Tu verras les courbes qu'un réseau sait dessiner.

---

### ex04 — MNIST !

| | |
|---|---|
| Fichiers | tous les précédents (copies) et `train_mnist.py` |
| Évaluation | `python3 moulinette/moulinette.py module06 --long` |

`train_mnist.py` est un script qui :

1. Charge MNIST depuis `data/mnist/` (chemin relatif à ton fichier, voir le module 03).
2. Construit un réseau, par exemple `784 → 256 → ReLU → 10`.
3. Entraîne par mini-batch (taille 64, `lr = 0.1`, avec un shuffle à chaque epoch).
4. Affiche l'accuracy de test après chaque epoch.
5. **Sa dernière ligne affichée doit être exactement** `test accuracy: 0.XXXX` (4 décimales).

**Objectif : ≥ 97 %.** Avec une bonne implémentation, ça prend moins d'une minute.

**Pour aller plus loin :**
- Trace la courbe de loss.
- Affiche les images que ton réseau rate. Certaines sont vraiment illisibles, même pour un humain.
- Essaie 2 couches cachées, d'autres `lr`, d'autres tailles de batch. Peux-tu atteindre 98 % ?

---

```bash
python3 moulinette/moulinette.py module06 --long
```

**Prends une seconde pour réaliser :** tu viens de coder, sans aucune bibliothèque d'IA, un programme qui **lit l'écriture manuscrite**. Personne ne lui a dit à quoi ressemble un 7. Il l'a appris.

Au [module 07](module07.md), tu découvres l'outil qu'utilisent les pros. Tu vas voir qu'il fait exactement ce que tu viens de coder.
