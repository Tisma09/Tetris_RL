from stable_baselines3.common.vec_env import SubprocVecEnv
from tetris_game import TetrisGame
from train import train, play_ia
from dql_agent import DQLAgent
import os

if __name__ == "__main__": 
    req_train = input("Do you want to start the training? (y/n)")
    if req_train == "y":
        num_simulations = int(input("How many simulations?"))
        num_episodes = int(input("How many episodes per simulation?"))
        num_batches = 1
        freq = 3

        filepath = "policy/dql_agent.pth"
        if os.path.exists(filepath):
            print("Entrainement existant, chargement du fichier...")
            agent = DQLAgent(234, 5, filename=filepath, loading=True)
        else:
            print("Aucun entrainement existant, création d'un nouvel agent...")
            agent = DQLAgent(234, 5, filename=filepath, loading=False)

        ##########################
        ##### Entraînement #######
        ##########################

        print("Toutes les simulations sont terminées.")
    else:
        game = TetrisGame(ui=True)
        agent = DQLAgent(234, 5)
        play_ia(agent, game)
        print("End of the game")
