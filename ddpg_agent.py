import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

# todo, check hyperparam
TAU = 1e-2              # for soft update of target parameters
WEIGHT_DECAY = 0.0      # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed,
                 lr_a=1e-4, lr_c=1e-4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        # todo-1-note: add n_agent for this class to generalize to MA setting
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents=n_agents
        self.lr_a=lr_a
        self.lr_c=lr_c
        self.seed = random.seed(random_seed)
        self.t_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_a)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_c, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # todo, check memory is deleted
        # todo, check self.step is deleted

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return action #todo: to see if np.clip(-1,1) is needed

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, actions_pred_loc, next_actions_pred_tar):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print("state shape (in learn)", states.shape)
        # print("action shape (in learn)", actions.shape)
        # print("rewards shape (in learn)", rewards.shape)
        # print("next_states (in learn)", next_states.shape)
        # print("dones (in learn)", dones.shape)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions_pred_tar)
        # Compute Q targets for current states (y_i)
        # print("rewaerd shape: ", rewards.shape)
        # print("dones shape:", dones.shape)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        #print("shape Q_tar", Q_targets.shape)
        #print("shape Q_exp", Q_expected.shape)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        #
        actor_loss = -self.critic_local(states, actions_pred_loc).mean()

        # test = -self.critic_local(states, next_actions_pred)
        # print(test.shape)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        # todo, debug
        #actor_loss.backward(retain_graph=True) # not sure if this is correct?
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

