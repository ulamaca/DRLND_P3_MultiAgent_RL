import numpy as np
import random
import copy
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

# todo, check hyperparam

WEIGHT_DECAY_ACTOR = 0.0      # L2 weight decay
WEIGHT_DECAY_CRITIC = 0.0      # L2 weight decay

NOISE_START = 1.0
NOISE_END = 0.2
NOISE_REDUCTION = 0.999
EPISODES_BEFORE_TRAINING = 300


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed=10,
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


        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_a, weight_decay=WEIGHT_DECAY_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_c, weight_decay=WEIGHT_DECAY_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, states, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""

        if i_episode > EPISODES_BEFORE_TRAINING and self.noise_scale > NOISE_END:
            # self.noise_scale *= NOISE_REDUCTION
            self.noise_scale = NOISE_REDUCTION ** (i_episode - EPISODES_BEFORE_TRAINING)
        # else keep the previous value

        if not add_noise:
            self.noise_scale = 0.0

        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        # add noise
        actions += self.noise_scale * self.add_noise2()  # works much better than OU Noise process
        # actions += self.noise_scale*self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        # for MADDPG
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
        full_states, actor_full_actions, full_actions, agent_rewards, agent_dones, full_next_states, critic_full_next_actions = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get Q values from target models
        Q_target_next = self.critic_target(full_next_states, critic_full_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = agent_rewards + gamma * Q_target_next * (1 - agent_dones)
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(input=Q_expected,
                                 target=Q_target)  # target=Q_targets.detach() #not necessary to detach
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local.forward(full_states,
                                                actor_full_actions).mean()  # -ve b'cse we want to do gradient ascent
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def add_noise2(self):
        noise = 0.5*np.random.randn(1, self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise

    def targets_update(self, tau):
        """
        update actor/critic's target networks
        :return:
        """
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

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
        self.size=size
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
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size) # modify the original version which used random integer generator
        self.state = x + dx
        return self.state



    # def act_ou(self, state, add_noise=True):
    #     """Returns actions for given state as per current policy."""
    #     state = torch.from_numpy(state).float().to(device)
    #     self.actor_local.eval()
    #     with torch.no_grad():
    #         action = self.actor_local(state).cpu().data.numpy()
    #     self.actor_local.train()
    #     if add_noise:
    #         action += self.noise.sample()
    #     return np.clip(action, -1, 1)