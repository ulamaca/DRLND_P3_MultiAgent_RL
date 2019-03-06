import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_agent import Agent as DDPGAgent
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

# todo, check hyperparam
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.95          # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
LEARN_EVERY = 30        # updating frequency
UPDATES_PER_LEARN = 20  # learning steps per learning step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def xenv_to_mem(xenv):
    """output format of [batch_size, n_agents*dx]"""
    return xenv.reshape((-1, xenv.shape[0]*xenv.shape[1]))


def xenv_to_oi(xenv, ith_agent):
    """output format of [1, dx]"""
    return xenv[np.newaxis, ith_agent, :]


def xmem_to_oi(xmem, ith_agent, state_size):
    """output format of [batch_size, dx]"""
    start = ith_agent * state_size
    end=start+state_size
    if xmem.shape[0] == 1:
        return xmem[np.newaxis, start:end]
    elif xmem.shape[0] > 1:
        return xmem[:, start:end]
    else:
        raise ValueError("xmem has incorrect format")

class MADDPGAGENT():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed, lr_a=LR_ACTOR, lr_c=LR_CRITIC):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.n_agents = n_agents
        self.agents = dict([(i, None) for i in range(n_agents)])
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0

        for i in range(n_agents):
            self.agents[i] = DDPGAgent(state_size, action_size, n_agents, lr_a, lr_c, random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, xenv, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        self.memory.add(xenv_to_mem(xenv),
                        xenv_to_mem(action),
                        reward,
                        xenv_to_mem(next_state),
                        done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        # print("t_step: ", self.t_step)
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_LEARN):
                    experiences = self.memory.sample()
                    #print("current memory length", len(self.memory))
                    # print("learning happening")
                    self.learn(experiences, GAMMA)

    def single_agent_act(self, o_i, ith_agent, add_noise=True):
        """return actions of all agents"""
        return self.agents[ith_agent].act(o_i, add_noise=add_noise)

    def multi_agents_act(self, xenv, add_noise=True):
        """return actions of all agents"""
        actions=[]
        for ith_agent in range(self.n_agents):
            o_i=xenv_to_oi(xenv, ith_agent)
            actions.append(self.single_agent_act(o_i, ith_agent, add_noise=add_noise))
        actions = np.concatenate(actions, axis=0)
        # print("multi_agent_act shape:", actions.shape)
        return actions

    def reset(self):
        for i in range(self.n_agents):
            self.agents[i].reset()

    def learn(self, experiences, gamma):
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
        xmems, actions, rewards, next_xmems, dones = experiences



        actions_pred_loc=[]
        for ith_agent in range(self.n_agents):
            o_is =  xmem_to_oi(xmems, ith_agent, self.state_size)
            # print("o_is shape", o_is.shape)
            a_loc = self.agents[ith_agent].actor_local(o_is)
            actions_pred_loc.append(a_loc)

        next_actions_pred_targ=[]
        for ith_agent in range(self.n_agents):
            o_is =  xmem_to_oi(next_xmems, ith_agent, self.state_size)
            # print("o_is shape", o_is.shape)
            a_targ = self.agents[ith_agent].actor_local(o_is)
            next_actions_pred_targ.append(a_targ)

        #print("Mu(x') shape: ", next_actions_pred.shape)

        for ith_agent in range(self.n_agents):
            actions_pred_loc_i=[a if i==ith_agent else a.detach() for i, a in enumerate(next_actions_pred)]
            actions_pred_loc_i=torch.cat(actions_pred_loc_i, dim=-1) # the last dim = dim(A)
            single_ddpg_exps=(xmems, actions, rewards[:, ith_agent, np.newaxis], next_xmems, dones[:, ith_agent, np.newaxis])
            self.agents[ith_agent].learn(single_ddpg_exps, gamma,
                                         actions_pred_loc_i, next_actions_pred_targ) # action prediction



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

# note. this is the ReplayBuffer for MADDPG ONLY
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        # print("states shape (in memory)", states.shape)
        # print("actions shape (in memory)", actions.shape)
        # print("rewards shape (in memory)", rewards.shape)
        # print("dones shape (in memory)", dones.shape)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

