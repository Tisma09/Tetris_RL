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
import torch.multiprocessing as mp
from tetris_game import TetrisGame
from train import train
from dql_agent import DQLAgent
import torch
import os

def run_simulation(sim_id, shared_agent, filepath, lock):
    """ Fonction exécutée par chaque processus pour jouer et entraîner l'agent """
    print(f"Lancement de la simulation {sim_id}...")

    game = TetrisGame(ui=False)  # Désactiver l'UI pour les simulations parallèles
    agent = DQLAgent(234, 5)  # Créer un agent local

    # Charger le modèle partagé
    agent.policy.load_state_dict(shared_agent.state_dict())

    # Entraînement
    train(agent, game, num_episodes=5, ui=False)

    #lock

    print(f"Simulation {sim_id} terminée.")

if __name__ == "__main__":
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
