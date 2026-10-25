# Module 08 — Modèles de langage : du bigramme au mini-GPT

> *« Un modèle de langage ne fait qu'une chose : deviner le caractère suivant. Tout le reste en découle. »*

**Objectif :** comprendre comment fonctionnent ChatGPT ou Claude en construisant, étape par étape, des modèles de langage de plus en plus puissants. Le dernier sera un **Transformer**, l'architecture de GPT, entraîné sur *Les Misérables* de Victor Hugo.
**Durée estimée :** 1 à 2 semaines.
**Rendu :** `rendu/module08/exXX/`
**Prérequis :** PyTorch installé (module 07), `data/miserables.txt` présent (`./data/get_data.sh`).

---

## Le cours

### Qu'est-ce qu'un modèle de langage ?

Un modèle qui, à partir d'un début de texte, donne une **probabilité pour chaque prochain caractère** (ou mot, ou morceau de mot) possible :

```
"Jean Valjea"  →  P("n") = 0.97,  P("u") = 0.01,  ...
```

Pour **générer** du texte, on tire un caractère au hasard selon ces probabilités, on l'ajoute au texte, et on recommence. C'est exactement ce que fait ChatGPT, mot après mot (enfin, *token* après *token*). La seule différence est la taille : des milliards de paramètres, et une grosse partie d'Internet comme données.

Pour **entraîner**, rien de nouveau : c'est de la **classification** (module 04). Les classes sont les caractères possibles, et on utilise la cross-entropy.

### Étape 1 : le bigramme (compter)

Le modèle le plus simple : le prochain caractère ne dépend **que du caractère actuel**. On compte, dans un gros texte, combien de fois chaque paire (bigramme) apparaît :

```
mot "chat" → bigrammes : (.c) (ch) (ha) (at) (t.)
```

Le `.` est un caractère spécial qui marque le début et la fin d'un mot.

On range les comptes dans une matrice `N[i, j]` = nombre de fois où le caractère `j` suit le caractère `i`. On normalise chaque ligne pour obtenir des probabilités :

```
P[i, j] = N[i, j] / Σₖ N[i, k]
```

**Le lissage :** si un bigramme n'apparaît jamais, `P = 0` et `log(0) = -∞`. On ajoute donc 1 à toutes les cases avant de normaliser (`N + 1`), pour qu'aucun bigramme ne soit impossible.

**Évaluer :** la **negative log-likelihood** moyenne, `-mean(log P[bigrammes réels])`. C'est exactement la cross-entropy ! Plus elle est basse, mieux le modèle prédit le texte. Un modèle qui ne sait rien (uniforme sur 39 caractères) a une NLL de `ln(39) ≈ 3.66`.

### Étape 2 : le MLP avec embeddings (Bengio, 2003)

Le bigramme ne regarde qu'un seul caractère en arrière, c'est beaucoup trop peu. Et si on regardait les 3 derniers ? Avec des comptes, il faudrait une table de 39³ × 39 cases, presque toutes vides. L'idée, c'est d'utiliser un **réseau de neurones**.

**Les embeddings :** on ne peut pas mettre la lettre « e » directement dans un réseau. On donne à chaque caractère un **vecteur appris** de, disons, 10 nombres. C'est une simple table de correspondance : `nn.Embedding(vocab_size, 10)`, qui est une matrice `(vocab_size, 10)` dont on lit la ligne `i` pour le caractère `i`.

Au fil de l'entraînement, les caractères qui « se comportent pareil » (les voyelles entre elles, par exemple) finissent avec des vecteurs proches. Le modèle invente sa propre représentation.

```
contexte "cha"  →  [emb(c), emb(h), emb(a)]  →  concaténation (30)  →  Linear → tanh → Linear  →  logits (vocab)
```

### Étape 3 : l'attention (le cœur de GPT)

Le MLP a un contexte fixe et minuscule. On aimerait que le modèle puisse regarder **tout** le texte précédent, et décider lui-même **où regarder**.

**Self-attention :** chaque position `t` du texte produit trois vecteurs (par multiplication matricielle de son embedding `x`) :

- une **Query** `q = x @ Wq` : « qu'est-ce que je cherche ? »
- une **Key** `k = x @ Wk` : « qu'est-ce que je contiens ? »
- une **Value** `v = x @ Wv` : « qu'est-ce que je transmets si on me choisit ? »

Puis :

```
1. scores = Q @ Kᵀ / sqrt(head_size)    (T, T) : à quel point la position t s'intéresse à la position s
2. masque causal : scores[t, s] = -∞ si s > t       ← on ne peut pas regarder le FUTUR
3. poids = softmax(scores, sur chaque ligne)        ← chaque ligne somme à 1
4. sortie = poids @ V                                (T, head_size) : moyenne pondérée des values
```

**Intuition :** dans « *Jean Valjean prit le chandelier. Il* », pour prédire ce qui suit « Il », la position de « Il » envoie une query qui ressemble à « je cherche un sujet masculin ». La key de « Valjean » répond fort, et l'attention ramène son information.

**Pourquoi `/ sqrt(head_size)` ?** Sans ça, les produits scalaires deviennent grands quand la dimension augmente, la softmax sature (un seul poids à 1, les autres à 0) et les gradients meurent.

**Pourquoi le masque causal ?** Pendant l'entraînement, on prédit **toutes les positions en parallèle** (énorme gain de vitesse). Si la position 5 pouvait voir la position 6, elle tricherait en lisant la réponse. Le masque `torch.tril` (matrice triangulaire inférieure) l'en empêche.

**Multi-head :** on fait plusieurs attentions en parallèle (des « têtes »), chacune avec ses propres `Wq`, `Wk` et `Wv`, plus petites. Une tête peut suivre la grammaire, une autre les noms propres, etc. On concatène les sorties, puis une projection linéaire recombine le tout.

### Étape 4 : le Transformer

Un **bloc Transformer** :

```
x = x + MultiHeadAttention(LayerNorm(x))     ← les positions communiquent entre elles
x = x + MLP(LayerNorm(x))                    ← chaque position "réfléchit" seule
```

- **Les connexions résiduelles** (`x = x + ...`) : le gradient peut traverser directement des dizaines de couches sans s'éteindre. C'est indispensable pour empiler beaucoup de blocs.
- **LayerNorm** : normalise chaque vecteur (moyenne 0, std 1), comme ta standardisation, mais à l'intérieur du réseau. Ça stabilise l'entraînement.
- Le **MLP** est en général `Linear(C, 4C) → GELU → Linear(4C, C)`.

**Le modèle complet (GPT) :**

```
tokens (B, T)
  → token embedding (B, T, C)  +  position embedding (T, C)    ← sinon le modèle ne connaît pas l'ordre !
  → N blocs Transformer
  → LayerNorm
  → Linear(C → vocab_size)
  → logits (B, T, vocab)    : une prédiction du prochain caractère pour CHAQUE position
```

C'est **la même architecture** que GPT-2, GPT-3 et compagnie. Juste plus petite.

---

## Exercices

### ex00 — Bigramme

| | |
|---|---|
| Fichier | `bigram.py` |
| Autorisé | `numpy`, `re` |

- `extract_words(text)` : passe le texte en minuscules, extrait les mots avec `re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ]+", ...)`, garde ceux d'au moins 3 lettres, et renvoie la liste **triée** des mots **uniques**.
- `build_vocab(words)` : renvoie `(stoi, itos)`. `stoi` (*string to int*) est un dict caractère → indice, et `itos` l'inverse (une liste ou un dict). Le `.` a l'indice 0, puis viennent tous les caractères présents, **triés**.
- `bigram_counts(words, stoi)` : la matrice `N` des comptes (`np.ndarray` d'entiers, `(V, V)`), en encadrant chaque mot par des `.`.
- `bigram_probs(N, smoothing=1)` : la matrice des probabilités, lissée et normalisée par ligne.
- `sample_word(P, itos, rng)` : génère un mot. On part de `.`, on tire le caractère suivant avec `rng.choice(V, p=P[i])`, et on s'arrête au prochain `.`. Renvoie le mot sans les `.`.
- `nll(words, P, stoi)` : la negative log-likelihood moyenne sur tous les bigrammes des mots.

Dans le `__main__`, entraîne sur `data/miserables.txt` et génère 20 mots. Résultats attendus : environ 12 000 mots uniques, une NLL autour de 2.39, et des mots comme « ratechacorinévisonare »… Pas fameux. C'est normal : le modèle n'a qu'un caractère de mémoire.

---

### ex01 — MLP de langage

| | |
|---|---|
| Fichier | `mlp_lm.py` |
| Autorisé | `torch`, plus tes fonctions de l'ex00 |

- `build_vocab(words)` : comme l'ex00 (copie-la).
- `build_dataset(words, stoi, block_size=3)` : pour chaque mot, génère les paires (contexte de `block_size` caractères → caractère suivant). Le contexte commence rempli de `.` (indice 0). Renvoie `(X, Y)`, des tenseurs `torch.long` de formes `(N, block_size)` et `(N,)`.
  ```
  "chat" avec block_size=3 :
  ...  →  c
  ..c  →  h
  .ch  →  a
  cha  →  t
  hat  →  .
  ```
- `MLPLM(vocab_size, block_size=3, emb_dim=10, hidden=200)` : un `nn.Module` avec `nn.Embedding` → concaténation (`.view` ou `.flatten(1)`) → `Linear` → `tanh` → `Linear`. Prend `X` `(B, block_size)` et renvoie les logits `(B, vocab_size)`.
- `generate(model, itos, block_size=3, n=10, generator=None)` : génère `n` mots (même principe que l'ex00, avec `torch.multinomial(probs, 1, generator=generator)`). Renvoie une liste de str.

Dans le `__main__` :
1. Mélange les mots, puis sépare-les en **train (90 %)** et **validation (10 %)**. La validation mesure si le modèle généralise, comme le test set de MNIST.
2. Entraîne avec Adam par mini-batchs aléatoires de 64 (environ 5 000 pas).
3. **Objectif : une loss de validation sous 2.15**, nettement mieux que le bigramme. Les mots générés commencent à sonner français : « cruissant », « assant », « pétilli »…

**Expérimente :** `block_size = 5` ou `8`, des embeddings plus gros, plus de neurones. Surveille l'écart entre la loss de train et celle de validation : si le train baisse mais pas la validation, c'est de l'overfitting.

---

### ex02 — L'attention

| | |
|---|---|
| Fichier | `attention.py` |

- `causal_self_attention(x, Wq, Wk, Wv)` : `x` est `(B, T, C)` et chaque `W` est `(C, H)`. Applique les 4 étapes du cours et renvoie `(B, T, H)`. Le masque se construit avec `torch.tril(torch.ones(T, T))` et `masked_fill(..., float("-inf"))`.
- `MultiHeadAttention(n_embd, n_head, block_size)` : un `nn.Module`. Prend `(B, T, n_embd)` et renvoie `(B, T, n_embd)`. Chaque tête a une taille `n_embd // n_head`. Termine par une projection `nn.Linear(n_embd, n_embd)`.

Version simple pour `MultiHeadAttention` : une liste de têtes (`nn.ModuleList`), chacune avec ses `nn.Linear(n_embd, head_size, bias=False)` pour Q, K et V, puis `torch.cat` des sorties sur la dernière dimension.

Version pro (bonus) : un seul `nn.Linear(n_embd, 3 * n_embd)`, puis `.view` et `.transpose` pour traiter toutes les têtes en un seul produit matriciel.

La moulinette vérifie que ta position `t` ne voit **jamais** les positions `> t`.

---

### ex03 — Mini-GPT

| | |
|---|---|
| Fichiers | `attention.py` (copie) et `mini_gpt.py` |

Cette fois, on travaille au niveau **caractère** sur tout le texte : espaces, ponctuation, majuscules et retours à la ligne inclus.

- `Block(n_embd, n_head, block_size)` : un bloc Transformer (attention + MLP, avec LayerNorm et résiduels).
- `MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size)`, qui doit avoir :
  - `forward(idx, targets=None)` : `idx` est `(B, T)`. Renvoie `(logits, loss)`, avec `logits` de forme `(B, T, vocab_size)`, et `loss` qui vaut la cross-entropy si `targets` est fourni, `None` sinon. Pour `F.cross_entropy`, aplatis : `logits.view(B*T, -1)` et `targets.view(B*T)`.
  - `generate(idx, max_new_tokens)` : boucle `max_new_tokens` fois. On garde les `block_size` derniers tokens, on fait un forward, on prend les logits de la **dernière** position, puis softmax et `multinomial`, et on concatène. Renvoie `idx` agrandi.

Dans le `__main__` :
1. Charge `miserables.txt` et construis le vocabulaire des caractères (environ 100).
2. Encode tout le texte en un tenseur d'entiers, avec 90 % pour le train et 10 % pour la validation.
3. Pour faire un batch : tire 32 positions au hasard, `x = data[i : i+block_size]` et `y = data[i+1 : i+block_size+1]` (le même texte décalé d'un cran).
4. Entraîne avec `AdamW`, `lr = 1e-3`, avec par exemple `n_embd=128`, `n_head=4`, `n_layer=4` et `block_size=128`.
5. Affiche les loss de train et de validation tous les 250 pas, puis génère 500 caractères.

**Ce qui t'attend** (sur un CPU correct) :

| Pas | Temps | Loss val | Ce que ça génère |
|---|---|---|---|
| 0 | 0 s | ~4.4 | `xQ»;ç8kÉ…`, de la bouillie |
| 500 | ~1 min | ~1.9 | des mots, des espaces, de la ponctuation |
| 1 000 | ~2 min | ~1.6 | `Elle sorriait ouvraît agistra l'évêque de la pourcamarboeuille, en nuit.` |
| 2 000 | ~4 min | ~1.3 | des phrases presque grammaticales |
| 4 000 | ~8 min | ~1.25 | `L'ombre s'aille eût redemarquable son oeuvre, dit dont la royale pauvre profonde` |

**Objectif : une loss de validation sous 1.5.**

La moulinette vérifie les formes, la loss initiale (environ `ln(vocab_size)`, sinon ton initialisation est cassée) et l'absence de fuite du futur. La qualité du texte, c'est toi qui juges.

---

```bash
python3 moulinette/moulinette.py module08
```

## Le mot de la fin

Tu viens de construire, **de zéro**, la même architecture que ChatGPT. Ce qui sépare ton mini-GPT de GPT-4 ou de Claude :

1. **L'échelle :** des milliards de paramètres au lieu d'un million, et des milliers de milliards de tokens au lieu de 600 000 caractères.
2. **La tokenisation :** des morceaux de mots (BPE) au lieu de caractères.
3. **Le fine-tuning :** après l'entraînement « deviner la suite », on l'entraîne encore sur des conversations, avec des retours humains (RLHF), pour qu'il devienne un **assistant** et pas seulement un compléteur de texte.

Mais le principe, c'est **exactement** ce que tu as codé : des embeddings, de l'attention, des MLP, et une descente de gradient sur une cross-entropy.

**Prochaines étapes possibles :**
- Implémente un tokenizer BPE de zéro et entraîne ton GPT dessus.
- Les projets IA de 42 : `dslr`, `multilayer-perceptron`, `total-perspective-vortex`…
- Fine-tune un vrai petit modèle pré-entraîné (avec la bibliothèque Hugging Face `transformers`).
- Les GAN et les modèles de diffusion, pour générer des images.

Bravo. 🎓
