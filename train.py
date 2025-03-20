import pygame

import torch

from dql_agent import DQLAgent
from config import *



def train(agent, game_env, num_episodes=1000):
    """
    Entraîne un agent (DQL ou Gradient Learning) dans l'environnement Tetris.

    :param agent: L'agent à entraîner (DQLAgent ou GradientLearningAgent)
    :param game_env: L'environnement de jeu Tetris
    :param num_episodes: Nombre d'épisodes d'entraînement
    """
    for episode in range(num_episodes):
        state, reward, done = game_env.reset()  # Réinitialisation du jeu
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)  # L'agent choisit une action
            
            next_state, reward, done = game_env.step(action)  # Exécute l'action
            
            # Pour DQL : Stocke l'expérience, pour Gradient Learning : Entraîne directement
            if isinstance(agent, DQLAgent):
                agent.remember(state, action, reward, next_state, done)
            else:  # GradientLearningAgent entraîne immédiatement
                agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        # Pour DQLAgent : entraînement par mini-lots après chaque épisode
        if isinstance(agent, DQLAgent):
            agent.replay()

        print(f"Episode {episode+1}/{num_episodes} - Score: {total_reward}")

    pygame.quit()

