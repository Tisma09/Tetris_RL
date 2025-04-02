from stable_baselines3.common.vec_env import SubprocVecEnv
import os

from tetris_game import TetrisGame
from tetris_env import TetrisEnv
from dql_agent import DQLAgent
from train import train, train_multiprocess, play_ia
from plot_scores import plot_scores_list


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

        filepath = "policy/dql_agent.pth"
        if os.path.exists(filepath):
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
                         filename=filepath, 
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
                replay_frequency=50,
                num_batches=1
            )
        finally:
            env.close()
            
        print("Toutes les simulations sont terminées.")

        plot_scores_list(rewards_history)

        """     reward_history = np.array(rewards_history)
        num_cpu, num_episodes = reward_history.shape
        colors = plt.cm.rainbow(np.linspace(0, 1, num_cpu))
        plt.figure(figsize=(12, 6))

        # Tracer les récompenses pour chaque CPU
        for i in range(num_cpu):
            plt.plot(np.arange(1, num_episodes + 1), reward_history[i, :], 
                    label=f'CPU {i}', color=colors[i], alpha=0.7)

        # Labels et titre
        plt.xlabel("Numéro d'épisode")
        plt.ylabel("Récompense")
        plt.title("Historique des récompenses par CPU")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()"""

                
        ##########################

    else:
        game = TetrisGame(ui=True)
        agent = DQLAgent(234, 5)
        play_ia(agent, game)
        print("End of the game")
