from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np

from tetris_game import TetrisGame
from tetris_env import TetrisEnv
from dql_agent import DQLAgent
from train import train_multiprocess, play_ia
from plot_scores import create_dirs_logs, create_files_scores, extract_list


def make_env(rank):
    def _init():
        env = TetrisEnv()
        return env
    return _init

if __name__ == "__main__": 
    req_train = input("Do you want to start the training? (y/n)")
    if req_train == "y":
        
        ##########################
        #####     Agent    #######
        ##########################

        train_foldername, train_filename = 'policy', 'dql_agent.pth'

        os.makedirs(train_foldername, exist_ok=True)
        train_filepath = os.path.join(train_foldername, train_filename)
        
        if os.path.exists(train_filepath):
            print("Entrainement existant, chargement du fichier...")
            loading = True
        else:
            print("Aucun entrainement existant, création d'un nouvel agent...")
            loading = False
        

        ##################################
        # Param à ajuster : 
        ##################################
        agent = DQLAgent(state_size = 234, 
                         action_size = 5, 
                         filename=train_filepath, 
                         loading=loading, 
                         epsilon=1.0, 
                         epsilon_min=0.01, 
                         epsilon_decay=0.995, 
                         gamma=0.99, 
                         learning_rate=0.001, 
                         batch_size=32)

        
        ##########################
        ##### Entraînement #######
        ##########################
        num_cpu = int(input("How many simulations?")) # 10
        num_episodes = int(input("How many episodes per process?")) # 10
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
        # Lancer l'entraînement
        try :
            ##################################
            # Param à ajuster : 
            ##################################
            rewards_history = train_multiprocess(
                agent=agent,
                env=env,
                num_cpu=num_cpu,
                episodes_per_process=num_episodes, 
                replay_frequency=500,
                num_batches=1
            )
        finally:
            env.close()
            
        print("Toutes les simulations sont terminées.")

        rewards_history = np.array(rewards_history)

        logs_folderpath = create_dirs_logs(num_cpu, num_episodes)
        create_files_scores(logs_folderpath, rewards_history)

        print("Affichage des scores")
        # Affichage des scores
        extract_list(rewards_history)

    else:
        game = TetrisGame(ui=True)
        agent = DQLAgent(234, 5)
        play_ia(agent, game)
        print("End of the game")
