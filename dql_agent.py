import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQLAgent:
    def __init__(self, state_size, action_size, filename=None, loading=False,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99,
                 learning_rate=0.001, batch_size=32, max_memory_size=2000):
        # Taille état et nombre d'actions possibles
        self.state_size = state_size
        self.action_size = action_size

        # Taux d'exploration
        self.epsilon = epsilon 
        # Exploration minimale
        self.epsilon_min = epsilon_min
        # Décroissance de l'exploration
        self.epsilon_decay = epsilon_decay
        # Facteur d'actualisation des récompenses futures
        self.gamma = gamma 

        # Paramètres d'apprentissage
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Mémoire tampon de rejeu
        self.memory = deque(maxlen=max_memory_size)
        self.remember_call = 0
        self.stop = False
        
        # Réseau de politique
        self.policy = self._build_policy()
        # Optimiseur pour l'ajustement des poids
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        # Fonction de perte (erreur quadratique moyenne)
        self.loss_fn = nn.MSELoss()

        self.filename = filename
        if loading :
            self.load_policy()

    def _build_policy(self):
        """
        Création du réseau de neurones
        """
        policy = nn.Sequential(
            nn.Linear(self.state_size, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return policy

    def act(self, state):
        """
        Choisir une action
        """
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        # Exploitation
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.policy(state)
        return torch.argmax(q_values).item()  
    

    def remember(self, state, action, reward, next_state, done):
        """
        Sauvegarder l'état en mémoire
        """
        self.memory.append((state, action, reward, next_state, done))
        self.remember_call += 1


    def replay(self, num_batches=1):
        """
        Effectuer plusieurs étapes de rejeu pour mieux utiliser les données accumulées.

        num_batches: Nombre de lots à traiter en une étape de rejeu.
        """
        if len(self.memory) < self.batch_size:
            print("Mémoire inférieure à la taille du lot")
            return
        if len(self.memory) < self.memory.maxlen:
            print("Attente que la mémoire soit pleine...")
            return

        for _ in range(num_batches):
            # Sample a batch from memory
            batch = random.sample(self.memory, self.batch_size)

            for state, action, reward, next_state, done in batch:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                target = reward
                if not done:
                    target = reward + self.gamma * torch.max(self.policy(next_state)).item()

                target_f = self.policy(state)
                target_f[0][action] = target

                # Perform gradient descent step
                self.optimizer.zero_grad()
                loss = self.loss_fn(target_f, self.policy(state))
                loss.backward()
                self.optimizer.step()

        # Reduce epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #########################################
    ####            Sauvegarde          #####
    #########################################

    def save_policy(self):
        """
        Sauvegarder dans un fichier pth
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.filename)
        print(f"Politique sauvegardée sous '{self.filename}'")

    def load_policy(self):
        """
        Charger un fichier pth
        """
        try:
            checkpoint = torch.load(self.filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            print(f"Politique chargée depuis '{self.filename}'")
        except FileNotFoundError:
            print(f"Erreur : fichier '{self.filename}' introuvable.")
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
