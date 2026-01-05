#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>



/*
 * COMPILATION :
 *   gcc -O3 -mavx512f matrix_mult_avx512.c -o matrix_mult_avx512
 * 
 * EXÉCUTION AVEC INTEL SDE (si pas d'AVX-512 natif) :
 *   ./sde64 -icl -- ./matrix_mult_avx512
 * 
 * EXÉCUTION NORMALE :
 *   ./matrix_mult_avx512
 */




// Définition du seuil de tolérance pour la comparaison de flottants
#define EPSILON 1e-4

// ============================================================================
// PARTIE 1 : VERSION NAÏVE
// ============================================================================

/**
 * Multiplication de matrices naïve : C = A × B
 * A : matrice M × K
 * B : matrice K × N
 * C : matrice M × N (résultat)
 */
void matrix_mult_naive(const float *A, const float *B, float *C, 
                       int M, int K, int N) {
    // Initialisation de C à zéro
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    // Algorithme classique : C[i][j] = sum(A[i][k] * B[k][j])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// PARTIE 2 : VERSION OPTIMISÉE AVX-512
// ============================================================================

/**
 * Multiplication de matrices optimisée avec AVX-512 : C = A × B
 * A : matrice M × K
 * B : matrice K × N
 * C : matrice M × N (résultat)
 * 
 * Stratégie : Traiter les colonnes de C par blocs de 16 éléments
 * avec gestion des reliquats par masquage
 */
void matrix_mult_avx512(const float *A, const float *B, float *C, 
                        int M, int K, int N) {
    // Initialisation de C à zéro
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    // Nombre de blocs complets de 16 colonnes
    int num_full_blocks = N / 16;
    int remainder = N % 16;
    
    // Créer un masque pour les éléments restants (reliquats)
    __mmask16 remainder_mask = 0;
    if (remainder > 0) {
        remainder_mask = (__mmask16)((1 << remainder) - 1);
    }
    
    // Pour chaque ligne de A (et donc de C)
    for (int i = 0; i < M; i++) {
        // Traiter les blocs complets de 16 colonnes
        for (int j_block = 0; j_block < num_full_blocks; j_block++) {
            int j_start = j_block * 16;
            __m512 acc = _mm512_setzero_ps();
            
            // Accumulation sur K : C[i][j:j+16] += A[i][k] * B[k][j:j+16]
            for (int k = 0; k < K; k++) {
                // Broadcaster A[i][k] dans un vecteur de 16 éléments
                __m512 a_elem = _mm512_set1_ps(A[i * K + k]);
                
                // Charger 16 éléments consécutifs de B[k][j:j+16]
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j_start]);
                
                // Fused Multiply-Add : acc = a_elem * b_vec + acc
                acc = _mm512_fmadd_ps(a_elem, b_vec, acc);
            }
            
            // Stocker le résultat dans C[i][j:j+16]
            _mm512_storeu_ps(&C[i * N + j_start], acc);
        }
        
        // Traiter les colonnes restantes avec masquage
        if (remainder > 0) {
            int j_start = num_full_blocks * 16;
            __m512 acc = _mm512_setzero_ps();
            
            for (int k = 0; k < K; k++) {
                // Broadcaster A[i][k]
                __m512 a_elem = _mm512_set1_ps(A[i * K + k]);
                
                // Charger avec masque (éléments hors masque = 0)
                __m512 b_vec = _mm512_maskz_loadu_ps(remainder_mask, 
                                                      &B[k * N + j_start]);
                
                // FMA avec masque
                acc = _mm512_fmadd_ps(a_elem, b_vec, acc);
            }
            
            // Stocker avec masque (ne modifie que les éléments masqués)
            _mm512_mask_storeu_ps(&C[i * N + j_start], remainder_mask, acc);
        }
    }
}

// ============================================================================
// PARTIE 3 : FONCTIONS UTILITAIRES
// ============================================================================

/**
 * Vérifie si deux matrices peuvent être multipliées
 * Retourne 1 si compatible, 0 sinon
 */
int check_dimensions(int M, int K_A, int K_B, int N) {
    if (K_A != K_B) {
        printf(" ERREUR : Dimensions incompatibles pour la multiplication !\n");
        printf("   Nombre de colonnes de A (%d) ≠ Nombre de lignes de B (%d)\n", 
               K_A, K_B);
        return 0;
    }
    return 1;
}

/**
 * Initialise une matrice avec des valeurs aléatoires entre 0 et 1
 */
void init_matrix_random(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * Initialise une matrice avec des valeurs séquentielles (pour debug)
 */
void init_matrix_sequential(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(i + 1);
    }
}

/**
 * Affiche une matrice (limité aux petites dimensions)
 */
void print_matrix(const char *name, const float *matrix, int rows, int cols) {
    printf("\n%s (%d×%d):\n", name, rows, cols);
    if (rows > 10 || cols > 10) {
        printf("  [Matrice trop grande pour l'affichage complet]\n");
        printf("  Premiers éléments: %.4f, %.4f, %.4f, ...\n", 
               matrix[0], matrix[1], matrix[2]);
        return;
    }
    
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

/**
 * Compare deux matrices et vérifie qu'elles sont identiques (à EPSILON près)
 * Retourne 1 si identiques, 0 sinon
 */
int compare_matrices(const float *C1, const float *C2, int rows, int cols) {
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(C1[i] - C2[i]);
        if (diff > EPSILON) {
            errors++;
            if (diff > max_error) {
                max_error = diff;
            }
            if (errors <= 5) { // Afficher seulement les 5 premières erreurs
                printf("  Différence à [%d,%d]: naïve=%.6f, AVX-512=%.6f, diff=%.6f\n",
                       i / cols, i % cols, C1[i], C2[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("ÉCHEC : %d différences trouvées (erreur max: %.6f)\n", 
               errors, max_error);
        return 0;
    }
    
    printf("SUCCÈS : Les matrices sont identiques (tolérance: %.0e)\n", EPSILON);
    return 1;
}

/**
 * Mesure le temps d'exécution d'une fonction de multiplication
 */
double benchmark(void (*mult_func)(const float*, const float*, float*, int, int, int),
                 const float *A, const float *B, float *C,
                 int M, int K, int N, int num_iterations) {
    clock_t start = clock();
    
    for (int iter = 0; iter < num_iterations; iter++) {
        mult_func(A, B, C, M, K, N);
    }
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC / num_iterations;
}

// ============================================================================
// PARTIE 4 : TESTS
// ============================================================================

void test_small_matrix() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 1 : Petite matrice (3×4) × (4×5)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int M = 3, K = 4, N = 5;
    
    // Vérification des dimensions
    if (!check_dimensions(M, K, K, N)) return;
    
    float A[12] = {1,2,3,4,  5,6,7,8,  9,10,11,12};
    float B[20] = {1,2,3,4,5,  6,7,8,9,10,  11,12,13,14,15,  16,17,18,19,20};
    float C_naive[15];
    float C_avx512[15];
    
    print_matrix("Matrice A", A, M, K);
    print_matrix("Matrice B", B, K, N);
    
    matrix_mult_naive(A, B, C_naive, M, K, N);
    matrix_mult_avx512(A, B, C_avx512, M, K, N);
    
    print_matrix("C (naïve)", C_naive, M, N);
    print_matrix("C (AVX-512)", C_avx512, M, N);
    
    printf("\nVérification : ");
    compare_matrices(C_naive, C_avx512, M, N);
}

void test_exact_16_columns() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 2 : Matrice avec exactement 16 colonnes (10×8) × (8×16)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int M = 10, K = 8, N = 16;
    
    if (!check_dimensions(M, K, K, N)) return;
    
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_naive = (float*)malloc(M * N * sizeof(float));
    float *C_avx512 = (float*)malloc(M * N * sizeof(float));
    
    srand(42);
    init_matrix_random(A, M, K);
    init_matrix_random(B, K, N);
    
    matrix_mult_naive(A, B, C_naive, M, K, N);
    matrix_mult_avx512(A, B, C_avx512, M, K, N);
    
    printf("Vérification : ");
    compare_matrices(C_naive, C_avx512, M, N);
    
    free(A); free(B); free(C_naive); free(C_avx512);
}

void test_with_remainder() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 3 : Matrice avec reliquats (12×10) × (10×23)\n");
    printf("           [16 + 7 colonnes, test du masquage]\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int M = 12, K = 10, N = 23; // N=23 = 16 + 7 (reliquat)
    
    if (!check_dimensions(M, K, K, N)) return;
    
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_naive = (float*)malloc(M * N * sizeof(float));
    float *C_avx512 = (float*)malloc(M * N * sizeof(float));
    
    srand(123);
    init_matrix_random(A, M, K);
    init_matrix_random(B, K, N);
    
    matrix_mult_naive(A, B, C_naive, M, K, N);
    matrix_mult_avx512(A, B, C_avx512, M, K, N);
    
    printf("Vérification : ");
    compare_matrices(C_naive, C_avx512, M, N);
    
    free(A); free(B); free(C_naive); free(C_avx512);
}

void test_large_matrix() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 4 : Grande matrice (256×512) × (512×1024)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int M = 256, K = 512, N = 1024;
    
    if (!check_dimensions(M, K, K, N)) return;
    
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_naive = (float*)malloc(M * N * sizeof(float));
    float *C_avx512 = (float*)malloc(M * N * sizeof(float));
    
    srand(456);
    init_matrix_random(A, M, K);
    init_matrix_random(B, K, N);
    
    printf(" Benchmark naïve...\n");
    double time_naive = benchmark(matrix_mult_naive, A, B, C_naive, M, K, N, 1);
    printf("   Temps : %.4f secondes\n", time_naive);
    
    printf(" Benchmark AVX-512...\n");
    double time_avx512 = benchmark(matrix_mult_avx512, A, B, C_avx512, M, K, N, 1);
    printf("   Temps : %.4f secondes\n", time_avx512);
    
    printf("\n Accélération : %.2fx\n", time_naive / time_avx512);
    
    printf("\nVérification : ");
    compare_matrices(C_naive, C_avx512, M, N);
    
    free(A); free(B); free(C_naive); free(C_avx512);
}

void test_incompatible_dimensions() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 5 : Dimensions incompatibles (4×5) × (3×6)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int M = 4, K_A = 5, K_B = 3, N = 6;
    
    if (!check_dimensions(M, K_A, K_B, N)) {
        printf(" Test réussi : Incompatibilité correctement détectée\n");
    }
}

void test_edge_cases() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  TEST 6 : Cas limites\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    // Test 6a : Matrice 1×1
    printf("\n[6a] Matrice 1×1 × 1×1 :\n");
    int M1 = 1, K1 = 1, N1 = 1;
    float A1[1] = {5.0f};
    float B1[1] = {3.0f};
    float C1_naive[1], C1_avx512[1];
    
    matrix_mult_naive(A1, B1, C1_naive, M1, K1, N1);
    matrix_mult_avx512(A1, B1, C1_avx512, M1, K1, N1);
    printf("Résultat attendu: 15.0, naïve: %.1f, AVX-512: %.1f - ", 
           C1_naive[0], C1_avx512[0]);
    compare_matrices(C1_naive, C1_avx512, M1, N1);
    
    // Test 6b : Matrice avec N=1 colonne
    printf("\n[6b] Matrice 5×3 × 3×1 (une seule colonne) :\n");
    int M2 = 5, K2 = 3, N2 = 1;
    float *A2 = (float*)malloc(M2 * K2 * sizeof(float));
    float *B2 = (float*)malloc(K2 * N2 * sizeof(float));
    float *C2_naive = (float*)malloc(M2 * N2 * sizeof(float));
    float *C2_avx512 = (float*)malloc(M2 * N2 * sizeof(float));
    
    init_matrix_random(A2, M2, K2);
    init_matrix_random(B2, K2, N2);
    matrix_mult_naive(A2, B2, C2_naive, M2, K2, N2);
    matrix_mult_avx512(A2, B2, C2_avx512, M2, K2, N2);
    compare_matrices(C2_naive, C2_avx512, M2, N2);
    
    free(A2); free(B2); free(C2_naive); free(C2_avx512);
    
    // Test 6c : Matrice avec N=15 (juste avant 16)
    printf("\n[6c] Matrice 4×4 × 4×15 (reliquat de 15) :\n");
    int M3 = 4, K3 = 4, N3 = 15;
    float *A3 = (float*)malloc(M3 * K3 * sizeof(float));
    float *B3 = (float*)malloc(K3 * N3 * sizeof(float));
    float *C3_naive = (float*)malloc(M3 * N3 * sizeof(float));
    float *C3_avx512 = (float*)malloc(M3 * N3 * sizeof(float));
    
    init_matrix_random(A3, M3, K3);
    init_matrix_random(B3, K3, N3);
    matrix_mult_naive(A3, B3, C3_naive, M3, K3, N3);
    matrix_mult_avx512(A3, B3, C3_avx512, M3, K3, N3);
    compare_matrices(C3_naive, C3_avx512, M3, N3);
    
    free(A3); free(B3); free(C3_naive); free(C3_avx512);
}

// ============================================================================
// FONCTION PRINCIPALE
// ============================================================================

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  PROJET : MULTIPLICATION DE MATRICES AVEC AVX-512             ║\n");
    printf("║  Architectures et Modèles de Calcul                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    printf("\n Fonctionnalités implémentées :\n");
    printf("   ✓ Version naïve en C\n");
    printf("   ✓ Version optimisée AVX-512 avec intrinsics\n");
    printf("   ✓ Vérification de compatibilité des dimensions\n");
    printf("   ✓ Gestion des reliquats avec masquage _mm512_maskz_*\n");
    printf("   ✓ Suite de tests complète\n");
    printf("   ✓ Benchmarks de performance\n");
    
    // Exécution de tous les tests
    test_small_matrix();
    test_exact_16_columns();
    test_with_remainder();
    test_large_matrix();
    test_incompatible_dimensions();
    test_edge_cases();
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  TESTS TERMINÉS                                               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    return 0;
}
