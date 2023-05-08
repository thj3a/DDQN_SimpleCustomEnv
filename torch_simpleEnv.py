# utils imports
import pdb
import time
import numpy as np
import pandas as pd
import random
from collections import deque
from copy import deepcopy as dcopy
import os
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleEnv:
    def step(self, state, action):

          next_state = dcopy(state)
          if action == 0:
              next_state[0][0] -= 1
          else: 
              next_state[0][0] += 1
          
          done = False
          if next_state[0][0] == 0 or next_state[0][0] == 3:
              done = True

          if next_state[0][1] == 1 and next_state[0][0] == 1:
              next_state[0][1] = 0

          return next_state, done

    def calc_reward(self, state, next_state, done):
        if next_state[0][0] == 0:
            return -10
        elif next_state[0][0] == 3:
            return 10
        elif state[0][1] == 1 and next_state[0][0] == 1:
            return 3
        else:
            return 0

    def reset(self):
        done = False
        state = [[2,1]]
        return state, done

class ReplayMemory:
    def __init__(self, capacity=10_000):
      self.memory = deque([],maxlen=capacity)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, 
                 memory_size=10_000, 
                 batch_size=32, 
                 gamma=0.99,
                 exploration_max=1.0, 
                 exploration_min=0.01, 
                 exploration_decay=0.995,
                 learning_rate=0.005, 
                 tau=0.125, 
                 n_actions=2, 
                 n_inputs=2):
        super(DQN, self).__init__()
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.n_actions = n_actions
        self.n_inputs = n_inputs

        self.loss_history = []
        self.fit_count = 0

        self.nodes_queue = []

        self.lin1 = nn.Linear(n_inputs, 24)
        self.lin2 = nn.Linear(24, 24)
        self.lin3 = nn.Linear(24, n_actions)
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def get_action(self, state, should_explore=True):
        state = torch.Tensor(state)
        if should_explore:
            self.exploration_max *= self.exploration_decay
            self.exploration_max = max(self.exploration_min, self.exploration_max)
            if np.random.random() < self.exploration_max:
                return np.random.randint(0, self.n_actions)
        
        with torch.no_grad():
            # q_values = self.model.predict(state, verbose=0)
            q_values = self.forward(state)
            best_action = torch.argmax(q_values)
            return best_action

def replay():
    if memory.__len__() < model.batch_size:
        return 
    
    samples = memory.sample(model.batch_size)
    states, actions, rewards, next_states, dones = zip(*samples)
    targets = []

    for state, action, reward, next_state, done in samples:
        target = aux_model.forward(torch.Tensor(state))
        if done:
            target[0][action] = reward
        else:
            Q_future = torch.max(aux_model.forward(torch.Tensor(next_state)))
            target[0][action] = reward + Q_future * model.gamma
        
        targets.append(target)
    
    states = torch.cat([torch.Tensor(i) for i in states])
    targets = torch.cat(targets)

    loss = F.mse_loss(model.forward(states), targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss)

    return loss_history[-1]
    
def aux_train():
    aux_model.load_state_dict(model.state_dict())

episodes = 500
rewards = []
losses = []
r = []
inner_loss = []
loss_history = []
memory = ReplayMemory()

model = DQN()
aux_model = DQN()
optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

env = SimpleEnv()

for ep in range(episodes):
    state, done = env.reset()
    r = []
    l = []
    while not done:
        action = model.get_action(state)
        next_state, done = env.step(state, action)
        reward = env.calc_reward(state, next_state, done)
        
        memory.remember(dcopy(state), action, reward, dcopy(next_state), done)
        # print("State: ", state, "Action: ", action, "Reward: ", reward, "Next State: ", next_state)
        if memory.__len__() > 100:  
            loss = replay() 
            l.append(loss)

        state = dcopy(next_state)
        r.append(reward)
    
    losses.append(torch.mean(torch.Tensor(l)))
    rewards.append(torch.sum(torch.Tensor(r)))
    if memory.__len__() > 100 or ep % 10 == 0:  aux_train()
    print(f"Episode:{ep}, Ep Reward: {rewards[-1]}, Last Loss: {losses[-1]}, Exploration: {model.exploration_max}")

plt.plot(rewards, label="rewards")
plt.plot(losses, label="losses")
plt.legend()
plt.title("Torch Results")
plt.savefig("torch-results.png")