import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQLAgent:
    def __init__(self, state_size, action_size, filename=None, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99, learning_rate=0.001, batch_size=32):
        # Taille état et nombre d'actions possibles
        self.state_size = state_size
        self.action_size = action_size

        

        # Exploration rate
        self.epsilon = epsilon 
        # Exploration min
        self.epsilon_min = epsilon_min
        # Exploration diminution
        self.epsilon_decay = epsilon_decay
        # Pondération récompenses futures
        self.gamma = gamma 

        # Paramètres d'apprentissage
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Replay buffer
        self.memory = deque(maxlen=2000)
        
        # Politique
        self.policy = self.build_policy()
        # Ajustement poids
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        # Fonction de perte (erreur quadratique moyenne)
        self.loss_fn = nn.MSELoss()

        if filename :
            self.load_policy(filename)

    def build_policy(self):
        policy = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return policy

    def act(self, state):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Exploitation
        state = state.clone().detach().float().unsqueeze(0) 
        q_values = self.policy(state)
        return torch.argmax(q_values).item()  

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            print("Number of xp too weak")
            return
        
        # Sélection d'un échantillion aléatoire
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = state.clone().detach().float().unsqueeze(0) 
            next_state = next_state.clone().detach().float().unsqueeze(0) 
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

    def save_policy(self, filename="policy/dql_agent.pth"):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),          # Poids du modèle
            'optimizer_state_dict': self.optimizer.state_dict(),    # État de l'optimiseur
            'epsilon': self.epsilon                                 # Sauvegarde du taux d'exploration
        }
        torch.save(checkpoint, filename)
        print(f"Politique sauvegardé sous '{filename}'")

    def load_policy(self, filename="policy/dql_agent.pth"):
        try:
            checkpoint = torch.load(filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            print(f"Politique chargée depuis '{filename}'")
        except FileNotFoundError:
            print(f"Erreur : fichier '{filename}' introuvable.")
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
