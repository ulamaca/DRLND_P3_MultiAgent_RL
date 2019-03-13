import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# class FCNetwork(nn.Module):
#     def __init__(self, input_dim, hiddens, func=F.leaky_relu):
#         super(FCNetwork, self).__init__()
#         self.func=func
#
#         # Input Layer
#         fc_first = nn.Linear(input_dim, hiddens[0])
#         self.layers = nn.ModuleList([fc_first])
#         # Hidden Layers
#         layer_sizes = zip(hiddens[:-1], hiddens[1:])
#         self.layers.extend([nn.Linear(h1, h2)
#                             for h1, h2 in layer_sizes])
#         self.reset_parameters()
#
#
#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.weight.data.uniform_(*hidden_init(layer))
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = self.func(layer(x))
#         return x
#
#
# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed=10, hiddens=(256, 128)):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc_hidden = FCNetwork(state_size, hiddens, func=F.relu)
#         self.fc_actor = nn.Linear(hiddens[-1], action_size)
#         self.fc_actor.weight.data.uniform_(-3e-3, 3e-3) # last layer params reset
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         x=self.fc_hidden.forward(state)
#         x=self.fc_actor(x)
#         return torch.tanh(x)
#
#
# class Critic(nn.Module):
#     """Critic (Value) Model."""
#
#     def __init__(self, state_size, action_size, seed=10, hiddens_s=(256,), hiddens=(128,)):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fcs1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.hidden_s = FCNetwork(state_size, hiddens_s, func=F.relu)
#         self.hidden = FCNetwork(hiddens_s[-1]+action_size, hiddens, func=F.relu)
#         self.fc_critic = nn.Linear(hiddens[-1], 1)
#         self.fc_critic.weight.data.uniform_(-3e-3, 3e-3) # reset params
#
#
#     def forward(self, state, action):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         xs = self.hidden_s.forward(state)
#         x = torch.cat((xs, action), dim=1)
#         x = self.hidden.forward(x)
#         return self.fc_critic(x)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=10,
                 fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=10,
                 fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
