from tetris_game import TetrisGame
from train import train
from dql_agent import DQLAgent

if __name__ == "__main__":
    game = TetrisGame(ui=True)
    #game.run()

    agent = DQLAgent(234, 5, filename="policy/dql_agent.pth")
    #agent = DQLAgent(234, 5)
    train(agent, game, num_episodes=1, ui=True)

