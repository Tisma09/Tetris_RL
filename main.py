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

