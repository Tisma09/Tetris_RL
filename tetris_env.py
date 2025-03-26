import gym
from gym import spaces
from tetris_game import TetrisGame
import numpy as np

class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        
        self.game = TetrisGame(ui=False)
        self.action_space = spaces.Discrete(5) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(234,), dtype=np.float32)

    def step(self, action):
        state, reward, done = self.game.step(action)
        return state, reward, done, {}

    def reset(self):
        return self.game.reset()
