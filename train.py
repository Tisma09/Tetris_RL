import pygame
from dql_agent import DQLAgent
from config import *

def train(agent, game_env, num_episodes=1, ui=False, lock=None):
    """
    Entraîne un agent (DQL ou Gradient Learning) dans l'environnement Tetris.

    agent: L'agent à entraîner (DQLAgent ou GradientLearningAgent)
    game_env: L'environnement de jeu Tetris
    num_episodes: Nombre d'épisodes d'entraînement
    ui: Avec ou sans interface
    lock: Verrou pour synchroniser l'accès aux ressources partagées
    """
    for episode in range(num_episodes):
        # Réinitialisation jeu
        state = game_env.reset()
        done = False
        total_reward = 0

        while not done:
            # Choix action
            action = agent.act(state)

            # Exécution de l'action
            next_state, reward, done = game_env.step(action, ui=ui)

            # Pour DQL : Stocke l'expérience, pour Gradient Learning : Entraîne directement
            if isinstance(agent, DQLAgent):
                if lock:
                    lock.acquire()
                try:
                    agent.remember(state, action, reward, next_state, done)
                finally:
                    if lock:
                        lock.release()
            else:
                agent.train(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward


        print(f"Episode {episode+1}/{num_episodes} - Score: {total_reward}")
        print(f"Remember since last replay : {agent.remember_call}")
        
def check_for_replay(agent, lock, num_batches, freq=3):
    """
    Vérifie si l'agent DQL a suffisamment de nouvelle expériences pour entraînement.
    """
    while agent.stop == False:
        #print(f"Remember since last replay : {agent.remember_call}")
        if agent.remember_call > agent.memory.maxlen/freq:
            if lock:
                lock.acquire()
            try:
                print("Replay requested")
                agent.replay(num_batches=num_batches)
                agent.remember_call = 0
            finally:
                if lock:
                    lock.release()


def play_ia(agent, game_env):
    """
    Montre l'IA jouer.

    agent: L'agent
    game_env: L'environnement de jeu Tetris
    """
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
        next_state, reward, done = game_env.step(action, ui=True)  
        
        state = next_state
        total_reward += reward


    print(f"Score: {total_reward}")
    pygame.quit()

