import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=24, action_size=2, seed=42):
        """Initialize parameters and build model.
        
        ==========Params==========
            state_size (int): Dimension of each state (which has a length 24)
            action_size (int): Dimension of each action (which has a length of 2)
            seed (int): Random seed used for initializing network parameters. 
        ==========================
        The Actor network has 3 hidden layers with (256, 128, 64) hidden nodes, respectively
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Get an action from a given state."""
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
            
        x = F.relu(self.fc1(states))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size=24, action_size=2, seed=42):
        """Initialize parameters and build model.
        
        ==========Params==========
            state_size (int): Dimension of each state (which has a length 24 )
            action_size (int): Dimension of each action (which has a length of 2)
            seed (int): Random seed used for initializing network parameters. 
        ==========================
        Similar to the Actor network, the Critic network has 3 hidden layers with (256, 128, 64) 
        hidden nodes, respectively.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256+action_size, 128)
        self.fc3 = nn.Linear(128, 64)        
        self.fc4 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Estimate the Q-value given a pair of (state, action)."""
      
        xs = F.relu(self.fcs1(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
