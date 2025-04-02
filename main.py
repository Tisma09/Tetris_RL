from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np

from tetris_game import TetrisGame
from tetris_env import TetrisEnv
from dql_agent import DQLAgent
from train import train, train_multiprocess, play_ia
from plot_scores import create_dirs_logs, create_files_scores, extract_list


def make_env(rank):
    def _init():
        env = TetrisEnv()
        return env
    return _init

if __name__ == "__main__": 
    req_train = input("Voulez vous recommencer l'entrainement ? (y/n)")
    if req_train == "y":
        
        ##########################
        #####     Agent    #######
        ##########################

        train_foldername, train_filename = 'policy', 'dql_agent_40.pth'

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
        agent = DQLAgent(state_size = 216, 
                         action_size = 40, 
                         filename=train_filepath, 
                         loading=loading, 
                         epsilon=1.0, 
                         epsilon_min=0.01, 
                         epsilon_decay=0.9995,
                         gamma=0.99, 
                         learning_rate=0.001, 
                         batch_size=32,
                         max_memory=1000)

        
        ##########################
        ##### Entraînement #######
        ##########################
        num_cpu = int(input("Combien de simulations ?")) 
        num_episodes = int(input("Combien d'episode par simulation ?")) 
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
        # Lancer l'entraînement
        """
        game_env = TetrisGame(ui=True)
        train(agent, game_env, num_episodes=num_episodes, num_batches=1, ui=True)
        """
        
        try :
            ##################################
            # Param à ajuster : 
            ##################################
            rewards_history = train_multiprocess(
                agent=agent,
                env=env,
                num_cpu=num_cpu,
                episodes_per_process=num_episodes, 
                replay_frequency="episode",
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
        print("Fin du jeu")
