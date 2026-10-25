#!/bin/bash
# Télécharge les datasets de la piscine (MNIST pour les modules 06-07, Les Misérables pour le module 08).
set -e
cd "$(dirname "$0")"

mkdir -p mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "mnist/$f.gz" ]; then
        echo "Téléchargement de $f..."
        curl -sSfL -o "mnist/$f.gz" "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
    fi
done

if [ ! -f miserables.txt ]; then
    echo "Téléchargement des Misérables (Victor Hugo, tome I)..."
    curl -sSfL "https://www.gutenberg.org/cache/epub/17489/pg17489.txt" \
        | sed -n '/\*\*\* START OF/,/\*\*\* END OF/p' | sed '1d;$d' | tr -d '\r' > miserables.txt
fi

echo "OK : données prêtes dans $(pwd)"
