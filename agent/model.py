import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layers(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor model. """

    def __init__(self, state_size, action_size, seed, hidden_sizes=[400, 300]):
        """ Initialize parameters and define PyTorch model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        self.reset()

    def reset(self):
        self.fc1.weight.data.uniform_(*init_layers(self.fc1))
        self.fc2.weight.data.uniform_(*init_layers(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, -3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """ Critic model. """

    def __init__(self, state_size, action_size, seed, hidden_sizes=[400, 300]):
        """ Initialize parameters and define PyTorch model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.reset()

    def reset(self):
        self.fc1.weight.data.uniform_(*init_layers(self.fc1))
        self.fc2.weight.data.uniform_(*init_layers(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, -3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x