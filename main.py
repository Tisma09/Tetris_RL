"""
from tetris_game import TetrisGame
from train import train
from dql_agent import DQLAgent
import os

if __name__ == "__main__":
    game = TetrisGame(ui=True)
    #game.run()

    filepath = "policy/dql_agent.pth"

    if os.path.exists(filepath):
        print("Entrainement déjà existant, chargement du fichier")
        agent = DQLAgent(234, 5, filename=filepath)
    else:
        print("Aucun entrainement exisant, création d'un nouveau fichier")
        agent = DQLAgent(234, 5)
    train(agent, game, num_episodes=10  , ui=True)

"""
import os
import shutil
import torch.multiprocessing as mp
from tetris_game import TetrisGame
from train import train
from dql_agent import DQLAgent
from plot_scores import plot_scores

def run_simulation(sim_id, shared_agent, filepath, lock):
    """ Fonction exécutée par chaque processus pour jouer et entraîner l'agent """
    print(f"Lancement de la simulation {sim_id}...")

    game = TetrisGame(ui=False)  # Désactiver l'UI pour les simulations parallèles
    agent = DQLAgent(234, 5)  # Créer un agent local

    # Charger le modèle partagé
    agent.policy.load_state_dict(shared_agent.state_dict())

    # Définir le nom du fichier CSV pour cette simulation
    log_file = f'logs/scores_sim_{sim_id}.csv'

    # Entraînement
    train(agent, game, num_episodes=4, ui=False, log_file=log_file)

    print(f"Simulation {sim_id} terminée.")

if __name__ == "__main__":
    # Spécifier le chemin du dossier
    dossier_logs = 'logs'

    # Vérifier si le dossier existe et le supprimer
    if os.path.exists(dossier_logs):
        # Si le dossier contient des fichiers ou des sous-dossiers, utiliser shutil.rmtree
        shutil.rmtree(dossier_logs)
        print(f"Le dossier '{dossier_logs}' et son contenu ont été supprimés.")

    # Recréer le dossier 'logs'
    os.makedirs(dossier_logs, exist_ok=True)
    print(f"Le dossier '{dossier_logs}' a été recréé.")


    mp.set_start_method("spawn", force=True)  # Forcer le démarrage avec 'spawn' pour Windows
    num_simulations = 5  # Nombre de parties en parallèle

    filepath = "policy/dql_agent.pth"
    if os.path.exists(filepath):
        print("Entrainement existant, chargement du fichier...")
        agent = DQLAgent(234, 5, filename=filepath)
    else:
        print("Aucun entrainement existant, création d'un nouvel agent...")
        agent = DQLAgent(234, 5)

    # Partager le modèle entre les processus
    agent.policy.share_memory()

    # Création d'un verrou pour la mise à jour du fichier
    lock = mp.Lock()

    # Lancer plusieurs simulations en parallèle
    processes = []
    for i in range(num_simulations):
        p = mp.Process(target=run_simulation, args=(i, agent.policy, filepath, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Attendre la fin de toutes les simulations

    print("Toutes les simulations sont terminées.")

    plot_scores(num_simulations)