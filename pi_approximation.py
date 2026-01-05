"""
alcul de π par méthode Monte-Carlo avec architecture client/serveur MPI
Projet 2 - Méthode stochastique
"""

from mpi4py import MPI
import numpy as np
import time

# Configuration
PACKET_SIZE = 10_000_000  # 10 millions de tirages par paquet
PRECISION_THRESHOLD = 1e-6  # Seuil de convergence pour arrêter
MIN_SAMPLES = 10  # Nombre minimum d'échantillons avant de vérifier la convergence
MAX_DURATION = 10.0  # Durée maximale en secondes

def monte_carlo_sample(n_draws):
    """
    Calcule un échantillon Monte-Carlo pour estimer π/4
    
    Args:
        n_draws: nombre de tirages aléatoires
    
    Returns:
        (n_in, n_total): nombre de points dans le cercle et nombre total
    """
    # Génération de points aléatoires uniformes dans [0,1] × [0,1]
    x = np.random.uniform(0, 1, n_draws)
    y = np.random.uniform(0, 1, n_draws)
    
    # Test si les points sont dans le cercle : x² + y² < 1
    distances_squared = x**2 + y**2
    n_in = np.sum(distances_squared < 1)
    
    return n_in, n_draws


def client_process(rank, comm):
    """
    Processus client : calcule des échantillons et les envoie au serveur
    
    Args:
        rank: rang du processus client
        comm: communicateur MPI
    """
    print(f"[Client {rank}] Démarrage")
    
    sample_count = 0
    
    while True:
        # Calcul d'un échantillon de 10 millions de tirages
        n_in, n_total = monte_carlo_sample(PACKET_SIZE)
        sample_count += 1
        
        # Estimation locale de π
        pi_local = 4.0 * n_in / n_total
        
        print(f"[Client {rank}] Échantillon #{sample_count}: "
              f"N_in={n_in:,}, N_total={n_total:,}, π≈{pi_local:.6f}")
        
        # Envoi des résultats au serveur (rang 0)
        comm.send((n_in, n_total), dest=0, tag=rank)
        
        # Réception de la commande du serveur
        command = comm.recv(source=0, tag=rank)
        
        if command == "STOP":
            print(f"[Client {rank}] Arrêt reçu. Total échantillons calculés: {sample_count}")
            break
        elif command == "CONTINUE":
            continue
        else:
            print(f"[Client {rank}] Commande inconnue: {command}")
            break


def server_process(comm, num_clients):
    """
    Processus serveur : collecte les résultats et coordonne les clients
    
    Args:
        comm: communicateur MPI
        num_clients: nombre de processus clients
    """
    print(f"[Serveur] Démarrage avec {num_clients} clients")
    print(f"[Serveur] Taille des paquets: {PACKET_SIZE:,} tirages")
    print(f"[Serveur] Durée maximale: {MAX_DURATION} secondes")
    print("-" * 70)
    
    # Accumulateurs globaux
    total_n_in = 0
    total_n_total = 0
    pi_estimates = []
    
    start_time = time.time()
    active_clients = num_clients
    samples_received = 0
    
    # Dictionnaire pour suivre les clients actifs
    client_status = {i: True for i in range(1, num_clients + 1)}
    
    while active_clients > 0:
        # Réception d'un résultat d'un client (n'importe lequel)
        status = MPI.Status()
        n_in, n_total = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        
        client_rank = status.Get_source()
        samples_received += 1
        
        # Mise à jour des accumulateurs
        total_n_in += n_in
        total_n_total += n_total
        
        # Estimation courante de π
        pi_estimate = 4.0 * total_n_in / total_n_total
        pi_estimates.append(pi_estimate)
        
        elapsed_time = time.time() - start_time
        
        # Calcul de l'erreur (si on a assez d'échantillons)
        if len(pi_estimates) >= MIN_SAMPLES:
            recent_estimates = pi_estimates[-MIN_SAMPLES:]
            variation = max(recent_estimates) - min(recent_estimates)
        else:
            variation = float('inf')
        
        print(f"[Serveur] Échantillon #{samples_received} du client {client_rank}: "
              f"π≈{pi_estimate:.8f}, "
              f"variation={variation:.2e}, "
              f"tirages totaux={total_n_total:,}, "
              f"temps={elapsed_time:.2f}s")
        
        # Décision : continuer ou arrêter ?
        should_stop = False
        
        if elapsed_time >= MAX_DURATION:
            print(f"[Serveur] Durée maximale atteinte ({MAX_DURATION}s)")
            should_stop = True
        elif len(pi_estimates) >= MIN_SAMPLES and variation < PRECISION_THRESHOLD:
            print(f"[Serveur] Convergence atteinte (variation < {PRECISION_THRESHOLD})")
            should_stop = True
        
        # Envoi de la commande au client
        if should_stop and client_status[client_rank]:
            comm.send("STOP", dest=client_rank, tag=client_rank)
            client_status[client_rank] = False
            active_clients -= 1
            print(f"[Serveur] STOP envoyé au client {client_rank} "
                  f"({active_clients} clients restants)")
        else:
            comm.send("CONTINUE", dest=client_rank, tag=client_rank)
    
    # Résultats finaux
    final_time = time.time() - start_time
    final_pi = 4.0 * total_n_in / total_n_total
    error = abs(final_pi - np.pi)
    relative_error = error / np.pi * 100
    
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Valeur estimée de π    : {final_pi:.10f}")
    print(f"Valeur réelle de π     : {np.pi:.10f}")
    print(f"Erreur absolue         : {error:.2e}")
    print(f"Erreur relative        : {relative_error:.4f}%")
    print(f"Nombre total de tirages: {total_n_total:,}")
    print(f"Nombre d'échantillons  : {samples_received}")
    print(f"Nombre de clients      : {num_clients}")
    print(f"Temps d'exécution      : {final_time:.2f} secondes")
    print(f"Débit                  : {total_n_total / final_time:,.0f} tirages/seconde")
    print("=" * 70)


def main():
    """
    Fonction principale : initialise MPI et lance serveur ou client
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Vérification : il faut au moins 2 processus (1 serveur + 1 client)
    if size < 2:
        if rank == 0:
            print("❌ ERREUR: Ce programme nécessite au moins 2 processus MPI.")
            print("   Usage: mpirun -np N python3 pi.py  (avec N ≥ 2)")
            print("   Exemple: mpirun -np 5 python3 pi.py  (1 serveur + 4 clients)")
        exit(1)
    
    # Initialisation du générateur aléatoire avec une graine différente par processus
    np.random.seed(rank * 1000 + int(time.time() * 1000) % 1000)
    
    if rank == 0:
        # Processus 0 = Serveur
        num_clients = size - 1
        server_process(comm, num_clients)
    else:
        # Processus 1, 2, 3, ... = Clients
        client_process(rank, comm)


if __name__ == "__main__":
    main()