from stable_baselines3.common.vec_env import SubprocVecEnv
import os

from tetris_game import TetrisGame
from tetris_env import TetrisEnv
from dql_agent import DQLAgent
from train import train, train_multiprocess, play_ia
from plot_scores import creation_repertoire_logs, creation_fichier_scores, plot_scores


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
        state_size = 234  
        action_size = 5

        filepath = "policy/dql_agent.pth"
        if os.path.exists(filepath):
            print("Entrainement existant, chargement du fichier...")
            agent = DQLAgent(state_size, action_size, filename=filepath, loading=True)
        else:
            print("Aucun entrainement existant, création d'un nouvel agent...")
            agent = DQLAgent(state_size, action_size, filename=filepath, loading=False)

        
        ##########################
        ##### Entraînement #######
        ##########################
        num_cpu = int(input("How many simulations?")) # 10
        num_episodes = int(input("How many episodes per process?")) # 10
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        num_batches = 1
        replay_frequency = 10
    
        # Lancer l'entraînement
        try :
            rewards_history = train_multiprocess(
                agent=agent,
                env=env,
                num_cpu=num_cpu,
                episodes_per_process=num_episodes, 
                replay_frequency=replay_frequency,
                num_batches=num_batches
            )
        finally:
            env.close()
            
        print("Toutes les simulations sont terminées.")

        
        # Optionnellement, tracez les récompenses pour voir la progression
        import matplotlib.pyplot as plt
        plt.plot(rewards_history)
        plt.title("Évolution des récompenses")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense totale")
        plt.show()
                
        ##########################

    else:
        game = TetrisGame(ui=True)
        agent = DQLAgent(234, 5)
        play_ia(agent, game)
        print("End of the game")
