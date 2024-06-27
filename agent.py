import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 24)
        self.linear2 = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Related to replay buffer
        self.memory = deque(maxlen=2000)
        
        # Discount factor gamma for related to updating Q values
        self.gamma = 0.95
        
        # Epsilon for decision making processes. Decaying epsilon, so that the longer is the experience, the less random the decisions are, since epsilon is smaller and smaller
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        
        # Adam optimizer for gradient descent
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Mean Squared Error Loss function
        self.criterion = nn.MSELoss()
        
    # Replay buffer
    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        
    def act(self, state):
        # Epsilon greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state)
        
        with torch.no_grad():
            act_values = self.model(state)
            
        # Returning maximum Q value
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        # Sampling from memory (experience buffer)
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, terminal in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            terminal = torch.FloatTensor([terminal])
            
            # Performing Q-Learning incremental update
            target = reward
            if not terminal:
                target = (reward + self.gamma * torch.max(self.model(next_state)[0]))
            
            target_f = self.model(state)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
    