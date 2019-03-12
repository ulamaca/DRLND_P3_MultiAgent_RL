import numpy as np
import random
from collections import namedtuple, deque
from maddpg_agent_ref import DDPG as DDPGAgent
import torch


# todo, check hyperparam
BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
LEARN_EVERY = 1         # updating frequency
UPDATES_PER_LEARN = 3   # learning steps per learning step
EPISODES_BEFORE_LEARNING = 300
TAU = 1e-3              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def xenv_to_xmem(xenv):
    """
        input format of [n_agents, ds] in np.array
        output format of [1, n_agents*ds] in np.array
    """
    return np.reshape(xenv, newshape=(-1))


def xenv_to_oi(xenv, ith_agent):
    """
        input format of [n_agents, ds] in np.array
        output format of [1, ds] in np.array
    """
    #return xenv[np.newaxis, ith_agent, :]
    return np.reshape(xenv[ith_agent,:], newshape=(1,-1))


def xmem_to_oi(xmem, ith_agent, state_size):
    """
        input format of [batch_size, dx], torch tensor
        output format of [batch_size, ds=dx/N], torch tensor
    """
    start = ith_agent * state_size
    end=start+state_size
    if xmem.shape[0] == 1:
        return xmem[:, start:end].unsqueeze(0)
    elif xmem.shape[0] > 1:
        return xmem[:, start:end]
    else:
        raise ValueError("xmem has incorrect format, should have at least 2 dimensions")

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
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0
        self.agents = dict([(i, None) for i in range(n_agents)])
        for i in range(self.n_agents):
            self.agents[i]=DDPGAgent(state_size, action_size, n_agents)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def reset(self):
        for i in range(self.n_agents):
            self.agents[i].reset()

    def step(self, state, action, reward, next_state, done, n_episode):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(xenv_to_xmem(state),
                        state,
                        action,
                        reward,
                        xenv_to_xmem(next_state),
                        next_state,
                        done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % LEARN_EVERY

        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE and n_episode > EPISODES_BEFORE_LEARNING:
                for _ in range(UPDATES_PER_LEARN):
                    for ith_agent in range(self.n_agents):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA, ith_agent)
                    self.multi_agents_target_update(tau=TAU)

    def multi_agents_act(self, states, n_episode, add_noise=True):
        """return actions of all agents"""
        actions = []
        for agent_id, agent in self.agents.items():
            o_i=xenv_to_oi(states, agent_id)
            action = agent.act(o_i, n_episode, add_noise)
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    # def multi_agents_act(self, states, n_episode, add_noise=True):
    #     """return actions of all agents"""
    #     actions = []
    #     for agent_id, agent in enumerate(self.agents):
    #         action = agent.act(np.reshape(states[agent_id, :], newshape=(1, -1)), n_episode, add_noise)
    #         #action = np.reshape(action, newshape=(1, -1)), not necessary
    #         actions.append(action)
    #     actions = np.concatenate(actions, axis=0)
    #     return actions

    def multi_agents_target_update(self, tau):
        for ith_agent in range(self.n_agents):
            self.agents[ith_agent].soft_update_all()

    def learn(self, samples, gamma, agent_no):
        # for learning MADDPG
        full_states, states, actions, rewards, full_next_states, next_states, dones = samples

        whole_action_dim=self.action_size*self.n_agents

        # todo, if it works, to see what is different between agnet_id and agent_no
        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)
        for agent_id in range(self.n_agents):
            agent_next_state = next_states[:, agent_id, :]
            critic_full_next_actions[:, agent_id, :] = self.agents[agent_id].actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, whole_action_dim)

        agent = self.agents[agent_no]
        agent_state = states[:, agent_no, :]
        actor_full_actions = actions.clone()  # create a deep copy
        actor_full_actions[:, agent_no, :] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, whole_action_dim)

        full_actions = actions.view(-1, whole_action_dim)

        agent_rewards = rewards[:, agent_no].view(-1, 1)  # gives wrong result without doing this
        agent_dones = dones[:, agent_no].view(-1, 1)  # gives wrong result without doing this
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards, \
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)


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
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        print(xs[0:2,24:29])
        print(states[0:2,1, :5])
        print(A)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_xs = torch.from_numpy(np.array([e.next_x for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (xs, states, actions, rewards, next_xs, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
