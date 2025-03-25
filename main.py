import torch.multiprocessing as mp
from tetris_game import TetrisGame
from train import train, play_ia, check_for_replay
from dql_agent import DQLAgent
import os

def run_simulation(agent, num_episodes, lock):
    """ Fonction exécutée par chaque processus pour jouer et entraîner l'agent """
    game = TetrisGame(ui=False)
    train(agent, game, num_episodes=num_episodes, ui=False, lock=lock)

if __name__ == "__main__": 
    req_train = input("Do you want to start the training? (y/n)")
    if req_train == "y":
        num_simulations = input("How many simulations?")
        num_episodes = input("How many episodes per simulation?")
        num_simulations = int(num_simulations)
        num_episodes = int(num_episodes)
        num_batches = 1
        freq = 3

        mp.set_start_method("spawn", force=True)  # Forcer le démarrage avec 'spawn' pour Windows
        filepath = "policy/dql_agent.pth"
        if os.path.exists(filepath):
            print("Entrainement existant, chargement du fichier...")
            agent = DQLAgent(234, 5, filename=filepath, loading=True)
        else:
            print("Aucun entrainement existant, création d'un nouvel agent...")
            agent = DQLAgent(234, 5, filename=filepath, loading=False)

        manager = mp.Manager()
        shared_agent = manager.Namespace()
        shared_agent.agent = agent  # Partage de l'agent
        # Entrainement en parallèle
        lock = mp.Lock()
        processes = []
        for _ in range(num_simulations):
            # Corrected argument order: lock is passed before num_simulations
            p = mp.Process(target=run_simulation, args=(shared_agent.agent, num_episodes, lock))
            processes.append(p)
            p.start()

        if isinstance(agent, DQLAgent) :
            replay_process = mp.Process(target=check_for_replay, args=(shared_agent.agent, lock, num_batches, freq))
            replay_process.start()

        for p in processes:
            p.join()

        if isinstance(agent, DQLAgent) :
            shared_agent.agent.stop = True
            replay_process.join()

        agent.save_policy()
        print("Toutes les simulations sont terminées.")
    else:
        game = TetrisGame(ui=True)
        agent = DQLAgent(234, 5)
        play_ia(agent, game)
        print("End of the game")
