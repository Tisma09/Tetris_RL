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
    num_cpu = 4  # Nombre de processus en parallèle
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Création de l'agent
    state_size = 234  # Taille de l'état du jeu Tetris
    action_size = 4   # Nombre d'actions possibles
    agent = DQLAgent(state_size, action_size)

    # Entraînement parallèle
    total_episodes = 1000
    episodes_per_process = total_episodes // num_cpu

    try:
        steps_since_replay = 0
        replay_frequency = 100  # Faire un replay tous les 100 steps
        
        for episode in range(episodes_per_process):
            states = env.reset()
            dones = [False] * num_cpu
            total_rewards = 0

            while not all(dones):
                actions = [agent.act(state) for state in states]
                next_states, rewards, dones, _ = env.step(actions)
                
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.remember(state, action, reward, next_state, done)
                    steps_since_replay += 1
                
                states = next_states
                total_rewards += reward
                
                # Replay moins fréquent
                if steps_since_replay >= replay_frequency and len(agent.memory) > agent.batch_size:
                    agent.replay(num_batches=4)  # Plus de batches par replay
                    steps_since_replay = 0

                

                if dones[0]:
                    print(f"Episode {episode}/{episodes_per_process} - Score: {total_rewards[0]}")
                if dones[1]:
                    print(f"Episode {episode}/{episodes_per_process} - Score: {total_rewards[0]}")
                if dones[2]:
                    print(f"Episode {episode}/{episodes_per_process} - Score: {total_rewards[0]}")
                if dones[3]:
                    print(f"Episode {episode}/{episodes_per_process} - Score: {total_rewards[0]}")

            print(f"Episode {episode}/{episodes_per_process}")
            if episode % 10 == 0:
                agent.save_policy()

    finally:
        env.close()
