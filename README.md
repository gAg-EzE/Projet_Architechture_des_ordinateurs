# Projet : Architecture des Ordinateurs et Optimisation

Ce dépôt contient des projets et des scripts de démonstration axés sur l'optimisation des performances, le calcul scientifique et la programmation bas niveau en lien avec l'architecture des ordinateurs.

## Contenu du Dépôt

* **`matrice_mult_avx512.c`** : Implémentation en C d'une multiplication de matrices optimisée à l'aide des instructions vectorielles SIMD AVX-512 (Advanced Vector Extensions). Ce script illustre l'exploitation du parallélisme au niveau des données pour maximiser les performances de calcul.
* **`pi_approximation.py`** : Script Python permettant d'approximer la valeur de $\pi$ (Pi)[cite: 1], utile pour des tests de performance algorithmique ou des comparaisons d'efficacité numérique.

## Prérequis et Compilation

### 1. Pour la partie C (AVX-512)
Pour compiler et exécuter le code en C, vous avez besoin d'un processeur compatible avec les instructions AVX-512 et d'un compilateur compatible (GCC ou Clang) :

```bash
# Compilation avec les options d'optimisation et d'activation d'AVX-512
gcc -O3 -mavx512f matrice_mult_avx512.c -o matrice_mult_avx512

# Exécution
./matrice_mult_avx512
