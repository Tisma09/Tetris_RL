import pygame
from config import *
import numpy as np

def train(agent, game_env, num_episodes=1, freq=3, num_batches=1, ui=False):
    """
    Entraîne un agent DQL dans l'environnement Tetris.

    agent: L'agent à entraîner DQLAgent
    game_env: L'environnement de jeu Tetris
    num_episodes: Nombre d'épisodes d'entraînement
    freq: frequence d'entrainement (tout les combien d'episodes)
    num_batches: Nombre de lot pare ntraînement
    ui: Avec ou sans interface
    """
    for episode in range(num_episodes):
        state = game_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = game_env.step(action, ui=ui)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}/{num_episodes} - Score: {total_reward}")
        if agent.remember_call > agent.memory.maxlen/freq:
            print("Replay requested")
            agent.replay(num_batches=num_batches)
            agent.remember_call = 0


def train_multiprocess(agent, env, num_cpu, episodes_per_process, replay_frequency=100, num_batches=1):
    """
    Entraîne un agent DQL dans plusieurs environnement Tetris.

    agent: L'agent à entraîner DQLAgent
    env: L'environnement gym du jeu Tetris
    episodes_per_process: Nombre d'épisodes d'entraînement par process
    replay_frequency: frequence d'entrainement (tout les combien d'episodes)
    num_batches: Nombre de lot pare ntraînement
    """
    steps_since_replay = 0
    completed_episodes = [0] * num_cpu
    all_episodes_rewards = [list() for _ in range(num_cpu)]
    
    while min(completed_episodes) < episodes_per_process:
        # Réinitialiser tous les environnements
        states = env.reset()
        episode_active = [True] * num_cpu
        episode_rewards = [0] * num_cpu
        
        while any(episode_active):
            # Préparer les actions pour tous les environnements
            actions = [0] * num_cpu 
            # Sélection action pour actif
            for i in range(num_cpu):
                if episode_active[i]:
                    actions[i] = agent.act(states[i])
            
            # Exécuter une étape dans tous les environnements
            next_states, rewards, dones, _ = env.step(actions)
            
            # Résultats pour chaque environnement
            for i in range(num_cpu):
                if episode_active[i]:
                    episode_rewards[i] += rewards[i]
                    if dones[i]:
                        # Episode terminé pour cet environnement
                        episode_active[i] = False
                        completed_episodes[i] += 1
                        all_episodes_rewards[i].append(episode_rewards[i])
                        
                        print(f"Env {i} - Episode {completed_episodes[i]}/{episodes_per_process} - Score: {episode_rewards[i]:.2f}")
                    else:
                        # Stocker l'expérience en mémoire
                        agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                        states[i] = next_states[i]
                        steps_since_replay += 1
            
            # Apprentissage sur la mémoire
            if replay_frequency is int and steps_since_replay >= replay_frequency :
                agent.replay(num_batches=num_batches)
                steps_since_replay = 0
        
        if replay_frequency is str :
            agent.replay(num_batches=num_batches)
        
        # Affichage de la progression
        current_episode = min(completed_episodes)
        print(f"Complétion générale: Episode {current_episode}/{episodes_per_process}")
        print(f"Epsilon : {agent.epsilon}")
        
        # Sauvegarde périodique du modèle
        save_frequency = 10  # Fréquence de sauvegarde
        if current_episode % save_frequency == 0:
            print(f"Sauvegarde à l'épisode {current_episode}")
            agent.save_policy()
            
            # Afficher les performances moyennes sur les derniers épisodes
            print_frequency = 20  # Fréquence d'affichage
            if all(len(episodes) >= print_frequency for episodes in all_episodes_rewards):
                recent_rewards = [episodes[-print_frequency:] for episodes in all_episodes_rewards]  # Prendre les x derniers épisodes de chaque CPU
                avg_rewards = [np.mean(episodes) for episodes in recent_rewards]  # Moyenne par CPU
                print(f"Score moyen sur les 20 derniers épisodes par environnement: {float(np.mean(avg_rewards))}")

    print(f"Sauvegarde finale à l'épisode {current_episode}")
    agent.save_policy()

    return all_episodes_rewards


def play_ia(agent, game_env):
    """
    Montre l'IA jouer.

    agent: L'agent
    game_env: L'environnement de jeu Tetris
    """
    # Réinitialisation jeu
    state = game_env.reset()  
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

