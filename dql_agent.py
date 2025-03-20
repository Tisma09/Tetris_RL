import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQLAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99, learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=2000)
        
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def act(self, state):
        # Exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        state = state.clone().detach().float().unsqueeze(0) 
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploitation

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Randomly sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = state.clone().detach().float().unsqueeze(0) 
            next_state = next_state.clone().detach().float().unsqueeze(0) 
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f[0][action] = target

            # Perform gradient descent step
            self.optimizer.zero_grad()
            loss = self.loss_fn(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        # Reduce epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
