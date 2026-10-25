# Module 07 — PyTorch

> *« Tout ce que tu as codé dans les modules 05 et 06, en 30 lignes. »*

**Objectif :** apprendre PyTorch, le framework utilisé par quasiment toute la recherche et l'industrie (ChatGPT, Claude, Stable Diffusion…), puis construire un **réseau convolutif** qui atteint **99 %** sur MNIST.
**Durée estimée :** 3 à 4 jours.
**Rendu :** `rendu/module07/exXX/`

---

## Installation

PyTorch est gros (environ 200 Mo en version CPU, largement suffisant pour cette piscine) :

```bash
source ~/ia-env/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Vérifie :

```bash
python3 -c "import torch; print(torch.__version__)"
```

---

## Le cours

### Les tenseurs : NumPy++

Un `torch.Tensor`, c'est un `np.ndarray` qui sait en plus deux choses : **calculer ses gradients** et **tourner sur GPU**.

```python
import torch

t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
t.shape            # torch.Size([2, 2])
t @ t              # produit matriciel, comme NumPy
t.sum(dim=0)       # "dim" au lieu de "axis", même logique
t.mean(), t.max(), t.argmax(dim=1)
torch.zeros(3, 4), torch.randn(3, 4), torch.arange(10)

# Conversion NumPy <-> PyTorch
torch.from_numpy(np_array)          # garde le dtype (float64 en général)
torch.tensor(np_array, dtype=torch.float32)  # copie en float32, le standard en deep learning
t.numpy()
```

La syntaxe est à 95 % celle de NumPy. Quand tu ne sais pas, essaie le nom NumPy : ça marche souvent.

### Autograd : ton `Value`, en tenseur

```python
x = torch.tensor(3.0, requires_grad=True)   # "suis mes opérations"
y = x * x + x                                # le graphe se construit tout seul
y.backward()                                 # tri topologique + règle de la chaîne
print(x.grad)                                # tensor(7.)  ← comme ton test du module 05 !
```

C'est **exactement** ton module 05. `requires_grad=True` marque une feuille dont on veut le gradient, `.backward()` remonte le graphe et `.grad` contient le résultat. `.item()` convertit un tenseur à un seul élément en `float` Python.

### `nn.Module` : tes classes `Layer` et `MLP`, version pro

```python
from torch import nn

class MonReseau(nn.Module):
    def __init__(self):
        super().__init__()                 # obligatoire
        self.fc1 = nn.Linear(784, 128)     # ta classe Dense ! (W, b, init, backward inclus)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                  # seulement le forward : le backward est automatique
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MonReseau()
logits = model(X)                          # appelle forward
model.parameters()                         # tous les W et b, comme ton parameters()
```

Plus compact, avec `nn.Sequential` :

```python
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
```

### La boucle d'entraînement canonique

À connaître **par cœur**, elle est identique dans 99 % des projets :

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # ou Adam, voir plus bas
loss_fn = nn.CrossEntropyLoss()                           # softmax + CE en un (prend les LOGITS)

for epoch in range(n_epochs):
    model.train()                             # mode entraînement
    for X_batch, y_batch in batches:
        logits = model(X_batch)               # 1. forward
        loss = loss_fn(logits, y_batch)       # 2. loss
        optimizer.zero_grad()                 # 3. remettre les grads à 0  (ton zero_grad !)
        loss.backward()                       # 4. backward
        optimizer.step()                      # 5. W -= lr * dW pour tous les paramètres
```

Et pour évaluer :

```python
model.eval()                                  # mode évaluation
with torch.no_grad():                         # pas besoin de construire le graphe : plus rapide
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
```

### Adam : un meilleur optimiseur

La descente de gradient classique (SGD) utilise le même `lr` pour tous les paramètres. **Adam** adapte le pas de chaque paramètre selon l'historique de ses gradients (une sorte de moyenne mobile, avec de l'élan). Dans la pratique, il converge plus vite et demande moins de réglages. `lr = 1e-3` est un bon point de départ.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### Les réseaux convolutifs (CNN)

Ton MLP du module 06 a un défaut : il **aplatit** l'image. Pour lui, le pixel (0, 0) et le pixel (0, 1) sont aussi éloignés que le pixel (0, 0) et le pixel (27, 27). Il ne « voit » pas que l'image est en 2D.

**La convolution :** on fait glisser un petit **filtre** (par exemple 3×3) sur toute l'image. À chaque position, on calcule une somme pondérée des 9 pixels sous le filtre. Le filtre est un petit détecteur de motif (bord vertical, coin, courbe…), et **ses 9 poids sont appris**.

```
Image          Filtre 3x3       Carte de caractéristiques
┌─────────┐    ┌─────┐          ┌───────┐
│ . . . . │    │ w w w│          │ ■ □ □ │   ← là où le motif est présent,
│ . ■ ■ . │ *  │ w w w│    →     │ ■ ■ □ │     la valeur est grande
│ . ■ . . │    │ w w w│          │ □ □ □ │
└─────────┘    └─────┘          └───────┘
```

Avantages :
1. **Partage des poids** : le même filtre sert partout. Un « 7 » est reconnu qu'il soit en haut à gauche ou au centre.
2. **Peu de paramètres** : 9 poids par filtre, contre 784 par neurone dans un MLP.
3. **Hiérarchie** : la première couche détecte des bords, la deuxième des formes, la troisième des chiffres.

**Le max pooling** divise la taille par 2 en gardant le max de chaque carré 2×2. Ça résume l'information et rend le réseau tolérant aux petits décalages.

```python
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # (N,1,28,28) → (N,32,28,28)
nn.MaxPool2d(2)                                                       # (N,32,28,28) → (N,32,14,14)
nn.Flatten()                                                          # (N,32,14,14) → (N,6272)
```

Une architecture classique pour MNIST :

```
(N,1,28,28) → Conv(1→32) → ReLU → Pool → Conv(32→64) → ReLU → Pool → Flatten → Linear → ReLU → Linear(→10)
```

Calcule les shapes à chaque étape sur papier pour trouver la taille d'entrée du premier `Linear`.

---

## Exercices

### ex00 — Tenseurs et autograd

| | |
|---|---|
| Fichier | `tensors.py` |

- `to_tensor(a)` : convertit un `np.ndarray` en tenseur **float32**.
- `derivative(f, x)` : `x` est un float. Renvoie `f'(x)` (sous forme de float Python) en utilisant l'**autograd** de PyTorch, pas la différence numérique.
- `neuron_grad(x, w, b)` : renvoie `(∂/∂w, ∂/∂b)` de `sigmoid(w·x + b)` avec autograd. Compare avec ton module 02 : même résultat, zéro calcul à la main.

---

### ex01 — MLP en PyTorch

| | |
|---|---|
| Fichier | `mlp.py` |

- `MLP(sizes)` : un `nn.Module`. `MLP([784, 128, 10])` donne `Linear(784, 128) → ReLU → Linear(128, 10)`. Pas de ReLU après la dernière couche.
- `train_one_epoch(model, X, y, optimizer, batch_size=64)` : une epoch complète avec shuffle (`torch.randperm`) et `nn.functional.cross_entropy`. Renvoie la **loss moyenne** (float).
- `evaluate(model, X, y)` : renvoie l'accuracy (float), dans un bloc `torch.no_grad()`.

Dans le `__main__`, refais le module 06 ex04 : charge MNIST (tu peux réutiliser ton `mnist.py`), entraîne avec Adam, et compare le nombre de lignes de code et le temps d'exécution. Tu devrais atteindre environ 98 %.

---

### ex02 — CNN

| | |
|---|---|
| Fichier | `cnn.py` |
| Évaluation | `python3 moulinette/moulinette.py module07 --long` |

- `CNN()` : un `nn.Module` qui prend des images `(N, 1, 28, 28)` et renvoie des logits `(N, 10)`. Il doit contenir au moins un `nn.Conv2d` et **moins de 500 000 paramètres**.
- Le `__main__` entraîne le modèle sur MNIST et affiche en **dernière ligne** `test accuracy: 0.XXXX`.

**Objectif : ≥ 99 %.** Compte environ 5 à 8 epochs avec Adam (`lr = 1e-3`) et des batchs de 128, soit 1 à 2 minutes sur un CPU récent.

**Si tu bloques vers 98,8 % :** plus d'epochs, un `nn.Dropout(0.25)` avant les `Linear` (on éteint au hasard 25 % des neurones pendant l'entraînement, ce qui force le réseau à ne pas dépendre d'un seul neurone), ou un learning rate qui diminue au fil des epochs (`torch.optim.lr_scheduler.StepLR`).

**Bonus :** visualise les 32 filtres de la première couche (`model.<ta_conv>.weight`) avec `plt.imshow`. Tu verras des détecteurs de bords apparaître tout seuls.

---

```bash
python3 moulinette/moulinette.py module07 --long
```

Tu maîtrises les images. Place au sujet du moment : le **texte**. Au [module 08](module08.md), tu construis un GPT.
