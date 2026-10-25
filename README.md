# Piscine IA

> Recrée une IA de zéro, de `print("Hello")` jusqu'à un mini-GPT qui écrit comme Victor Hugo.
> Tout en français, sans vidéo, façon 42 : un sujet, des exercices, une moulinette.

## Le parcours

| Module | Thème | Ce que tu sauras faire à la fin |
|---|---|---|
| [00](sujets/module00.md) | Python pour codeurs C | Écrire du Python propre sans réfléchir |
| [01](sujets/module01.md) | NumPy | Faire des calculs sur des matrices **sans aucune boucle** |
| [02](sujets/module02.md) | Les maths de l'IA | Dériver, calculer un gradient, faire une descente de gradient |
| [03](sujets/module03.md) | Régression linéaire | Ton premier modèle qui **apprend** à prédire un prix |
| [04](sujets/module04.md) | Classification | Prédire oui/non, puis choisir entre N classes |
| [05](sujets/module05.md) | Autograd | Coder le moteur de backpropagation qu'utilise PyTorch |
| [06](sujets/module06.md) | Réseau de neurones | Reconnaître des chiffres manuscrits à 97 %+, en NumPy pur |
| [07](sujets/module07.md) | PyTorch | Refaire tout ça en 10x moins de code, puis un CNN à 99 % |
| [08](sujets/module08.md) | Modèles de langage | Construire un mini-GPT et lui faire écrire du français |

Chaque module **dépend du précédent**. Ne saute rien : le module 06 n'a aucun sens si tu n'as pas fait les modules 02 et 05.

## Installation

Ton venv `~/ia-env` existe déjà avec NumPy, Matplotlib et Jupyter. À chaque nouveau terminal :

```bash
source ~/ia-env/bin/activate
```

Télécharge les données (MNIST et Les Misérables, environ 12 Mo) :

```bash
./data/get_data.sh
```

PyTorch ne sera nécessaire qu'au module 07 : l'installation est expliquée là-bas.

## Comment ça marche

```
IA/
├── sujets/          ← les sujets (cours + exercices), lis-les en entier
├── rendu/           ← TON code va ici
│   └── module00/
│       ├── ex00/hello.py
│       ├── ex01/basics.py
│       └── ...
├── moulinette/      ← le correcteur (n'y touche pas)
└── data/            ← les datasets
```

Pour te faire corriger :

```bash
python3 moulinette/moulinette.py module00
```

Tu peux aussi ne corriger qu'un seul exercice :

```bash
python3 moulinette/moulinette.py module00 ex03
```

Les règles sont les mêmes qu'à la piscine :

- La moulinette **s'arrête au premier KO**. Un ex00 raté, c'est 0 pour tout le module.
- Le nom des fichiers et des fonctions doit être **exactement** celui du sujet.
- Certains exercices interdisent des choses (boucles, fonctions, imports). La moulinette le vérifie.
- Les exercices d'entraînement long (MNIST, CNN…) ne tournent qu'avec `--long`.

## Les règles d'or

1. **Lis le cours en entier avant de coder.** Le sujet t'explique tout ce dont tu as besoin.
2. **Teste toi-même avant la moulinette.** Un `if __name__ == "__main__":` avec quelques `print` en bas de chaque fichier, c'est ton `main.c` de test.
3. **Ne copie pas de code d'Internet ni d'une IA.** Tu as le droit de chercher la doc (`help(np.sum)`, docs.python.org, numpy.org/doc), comme `man` en C. Mais si tu colles une solution, tu passes la moulinette sans rien comprendre, et le module suivant te le fera payer.
4. **Dessine.** Pour les maths et les formes de matrices, une feuille et un stylo valent mieux que 2 h devant l'écran.
5. **Bloqué plus de 30 min ?** Relis le cours, puis demande un **indice** (à un pote, à Claude…), pas la solution.

## Glossaire express

| Terme | En une phrase |
|---|---|
| **Modèle** | Une fonction avec des paramètres réglables, qui transforme une entrée en prédiction |
| **Paramètres / poids** | Les nombres que le modèle ajuste pour apprendre (souvent notés `W`, `b`, `θ`) |
| **Loss (perte)** | Un nombre qui mesure à quel point le modèle se trompe. On veut le minimiser |
| **Gradient** | La direction dans laquelle la loss augmente le plus vite. On va dans l'autre sens |
| **Descente de gradient** | Répéter : `param = param - lr * gradient` |
| **Learning rate (`lr`)** | La taille du pas à chaque mise à jour |
| **Epoch** | Un passage complet sur toutes les données d'entraînement |
| **Batch** | Un petit paquet d'exemples traité d'un coup |
| **Forward** | Calculer la prédiction à partir de l'entrée |
| **Backward / backprop** | Calculer les gradients, de la loss vers les paramètres |
| **Overfitting** | Le modèle apprend les données par cœur au lieu de généraliser |

## Et après ?

- Les projets de la branche IA de 42 (`ft_linear_regression`, `dslr`, `multilayer-perceptron`, `total-perspective-vortex`…) : après cette piscine, tu les dévoreras.
- Kaggle (kaggle.com) : des compétitions avec de vraies données.
- Le livre *Deep Learning* de Goodfellow, Bengio et Courville (gratuit en ligne, en anglais) pour la théorie complète.

Bon courage. Et rappelle-toi : une IA, ce n'est que des multiplications de matrices et des dérivées. Rien de magique.
