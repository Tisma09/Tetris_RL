import torch
import torch.nn as nn
import torch.optim as optim


class GradientLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state)).item()

        target_f = self.model(state)
        target_f[0][action] = target

        # Gradient descent update
        self.optimizer.zero_grad()
        loss = self.loss_fn(target_f, self.model(state))
        loss.backward()
        self.optimizer.step()
