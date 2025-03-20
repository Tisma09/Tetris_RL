from tetris_game import TetrisGame
from train import train
from dql_agent import DQLAgent

if __name__ == "__main__":
    game = TetrisGame()
    #game.run()

    agent = DQLAgent(234, 5)
    train(agent, game, 10)

