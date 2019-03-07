import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_agent import Agent as DDPGAgent
import torch


# todo, check hyperparam
BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
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
    return xenv.reshape((-1))


def xenv_to_oi(xenv, ith_agent):
    """
        input format of [n_agents, ds] in np.array
        output format of [1, ds] in np.array
    """
    #return xenv[np.newaxis, ith_agent, :]
    return np.reshape(xenv[ith_agent,:], newshape=(1,-1)


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
        self.memory.add(xenv,
                        xenv_to_xmem(xenv),
                        xenv_to_xmem(action),
                        reward,
                        next_state,
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
            action=self.single_agent_act(o_i, ith_agent, add_noise=add_noise)
            action=np.reshape(action, newshape=(1, -1))

            print("final action shape:", action.shape)
            print(A)

            actions.append(action)
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
        xs, xmems, states, actions, rewards, next_states, next_xmems, dones = experiences

        # 1. predict action from policy
        next_as_pred_targ = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)
        for jth_agent in range(self.n_agents): # j for another iterator for n_agents
            # 2. for next_as_pred_targ (for critic learning):
            next_o_is = next_states[:, ith_agent, :]
            # todo, a=(a1,a2,...mu(oi),...aN)
            next_as_pred_targ[:, jth_agent, :] = self.agents[jth_agent].actor_target(next_o_is) # todo, if to detach??, using target net for value only

        agent_state = states[:,ith_agent,:]
        as_pred_loc = actions.clone() #create a deep copy
        as_pred_loc[:,ith_agent,:] = self.agents[ith_agent].actor_local.forward(agent_state)
        as_pred_loc = as_pred_loc.view(-1, self.action_size*self.n_agents)

        # 2. learning
        single_ddpg_exps=(xmems, actions,
                          rewards[:, ith_agent].view(-1, 1),
                          next_xmems,
                          dones[:, ith_agent].view(-1, 1))

        self.agents[ith_agent].learn(single_ddpg_exps, gamma,
                                     as_pred_loc, next_as_pred_targ)


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
        self.experience = namedtuple("Experience", field_names=["x", "state", "action", "reward", "next_x", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, x, state, action, reward, next_x, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(x, state, action, reward, next_x, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        xs = torch.from_numpy(np.array([e.x for e in experiences if e is not None])).float().to(device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_xs = torch.from_numpy(np.array([e.next_x for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (xs, states, actions, rewards, next_xs, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

