from stable_baselines3.common.vec_env import SubprocVecEnv
from dql_agent import DQLAgent
from tetris_env import TetrisEnv
import threading

def make_env(rank):
    def _init():
        env = TetrisEnv()
        return env
    return _init

if __name__ == '__main__':
    num_cpu = 2  # Nombre de processus en parallèle
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Création de l'agent
    state_size = 234  
    action_size = 5
    num_batches = 4
    agent = DQLAgent(state_size, action_size)

    # Entraînement parallèle
    total_episodes = 4
    episodes_per_process = total_episodes // num_cpu

    try:
        steps_since_replay = 0
        replay_frequency = 100
        completed_episodes = [0] * num_cpu  # Compteur d'épisodes par environnement
        
        while min(completed_episodes) < episodes_per_process:
            states = env.reset()
            dones = [False] * num_cpu
            total_rewards = [0] * num_cpu
            active_envs = list(range(num_cpu))

            while active_envs:  # Continue tant qu'il y a des environnements actifs
                actions = [agent.act(states[i]) for i in active_envs]
                print("cc")
                next_states, rewards, dones, _ = env.step(actions)
                
                # Traiter chaque environnement actif
                print("cc2")
                new_active_envs = []
                for idx, env_idx in enumerate(active_envs):
                    if dones[idx]:
                        completed_episodes[env_idx] += 1
                        print(f"Env {env_idx} - Episode {completed_episodes[env_idx]}/{episodes_per_process} - Score: {total_rewards[env_idx]}")
                    else:
                        agent.remember(states[env_idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
                        total_rewards[env_idx] += rewards[idx]
                        states[env_idx] = next_states[idx]
                        new_active_envs.append(env_idx)
                        steps_since_replay += 1
                
                active_envs = new_active_envs

                print(active_envs)
                
                if steps_since_replay >= replay_frequency and len(agent.memory) > agent.batch_size:
                    agent.replay(num_batches=num_batches)
                    steps_since_replay = 0

            current_episode = min(completed_episodes)  # Utilise le moins avancé comme référence
            print(f"Episode : {current_episode}")
            if current_episode % 10 == 0:
                print(f"Sauvegarde à l'épisode {current_episode}")
                agent.save_policy()

    finally:
        env.close()
