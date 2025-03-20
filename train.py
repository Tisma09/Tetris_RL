import pygame

import torch

from dql_agent import DQLAgent
from config import *



def train(agent, game_env, num_episodes=1000, ui=False):
    """
    Entraîne un agent (DQL ou Gradient Learning) dans l'environnement Tetris.

    agent: L'agent à entraîner (DQLAgent ou GradientLearningAgent)
    game_env: L'environnement de jeu Tetris
    num_episodes: Nombre d'épisodes d'entraînement
    ui: Avec ou sans interface
    """

    for episode in range(num_episodes):
        # Réinitialisation jeu
        state, reward, done = game_env.reset()  
        done = False
        total_reward = 0
        pause = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause
            
            if pause :
                continue
            # Choix action
            action = agent.act(state)
            
            # Execute 
            next_state, reward, done = game_env.step(action, ui=ui)  
            
            # Pour DQL : Stocke l'expérience, pour Gradient Learning : Entraîne directement
            if isinstance(agent, DQLAgent):
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        # Pour DQLAgent : entraînement par mini-lots après chaque épisode
        if isinstance(agent, DQLAgent):
            agent.replay()

        print(f"Episode {episode+1}/{num_episodes} - Score: {total_reward}")

    agent.save_policy()
    if ui :
        pygame.quit()

