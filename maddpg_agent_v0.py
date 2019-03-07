import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_agent import Agent as DDPGAgent
import torch


# todo, check hyperparam
BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 2         # updating frequency
UPDATES_PER_LEARN = 1   # learning steps per learning step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def xenv_to_xmem(xenv):
    """
        input format of [n_agents, ds] in np.array
        output format of [1, n_agents*ds] in np.array
    """
    return xenv.reshape((-1, xenv.shape[0]*xenv.shape[1]))


def xenv_to_oi(xenv, ith_agent):
    """
        input format of [n_agents, ds] in np.array
        output format of [1, ds] in np.array
    """
    return xenv[np.newaxis, ith_agent, :]


def xmem_to_oi(xmem, ith_agent, state_size):
    """
        input format of [batch_size, dx], torch tensor
        output format of [batch_size, ds=dx/N], torch tensor
    """
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
        self.memory.add(xenv_to_xmem(xenv),
                        xenv_to_xmem(action),
                        reward,
                        xenv_to_xmem(next_state),
                        done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        # print("t_step: ", self.t_step)

        # print("learning happening")
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_LEARN):
                    for ith_agent in range(self.n_agents):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA, ith_agent)

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
        return actions

    def reset(self):
        for i in range(self.n_agents):
            self.agents[i].reset()

    def learn(self, experiences, gamma, ith_agent):
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

        #v3
        xmems, actions, rewards, next_xmems, dones = experiences

        # 1. predict action from policy
        as_pred_loc=[]
        next_as_pred_targ = []
        for jth_agent in range(self.n_agents): # j for another iterator for n_agents
            # 1. for as_pred_loc (for policy learning):
            o_is =  xmem_to_oi(xmems, jth_agent, self.state_size)
            # print("o_is shape", o_is.shape)
            a_pred_loc = self.agents[jth_agent].actor_local(o_is) # using local net for policy learning
            as_pred_loc.append(a_pred_loc)

            # 2. for next_as_pred_targ (for critic learning):
            next_o_is =  xmem_to_oi(next_xmems, jth_agent, self.state_size)
            # print("o_is shape", o_is.shape)
            # todo, a=(a1,a2,...mu(oi),...aN)
            next_a_pred_targ = self.agents[jth_agent].actor_target(next_o_is) # todo, if to detach??, using target net for value only
            next_as_pred_targ.append(next_a_pred_targ)
        next_as_pred_targ=torch.cat(next_as_pred_targ, dim=1)

        # 2. learning
        as_pred_loc_ith=[a if i==ith_agent else a.detach() for i, a in enumerate(as_pred_loc)]
        as_pred_loc_ith=torch.cat(as_pred_loc_ith, dim=1) # the last dim = dim(A)
        single_ddpg_exps=(xmems, actions,
                          rewards[:, ith_agent].view(-1, 1),
                          next_xmems,
                          dones[:, ith_agent].view(-1, 1))

        self.agents[ith_agent].learn(single_ddpg_exps, gamma,
                                     as_pred_loc_ith, next_as_pred_targ)


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

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

