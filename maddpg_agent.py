import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from collections import namedtuple, deque
from model import Actor, Critic

# Hyperparamerters:
BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 200        # minibatch size
GAMMA = 0.99            # discount factor
LEARN_EVERY = 2         # updating frequency
UPDATES_PER_LEARN = 4   # learning steps per learning step
STEPS_BEFORE_LEARNING = 5500
NOISE_START=1.0
NOISE_END=0.1
EXPLORATION_DECAY= 0.999 # 2/(t_max*n_episodes) let t_max~25, n_episodes=4000
L_FACTOR=80
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY_actor = 0.0  # L2 weight decay
WEIGHT_DECAY_critic = 0.0  # L2 weight decay
RANDOM_SEED=70
np.random.seed(seed=RANDOM_SEED) # to control noise sampling
# other possible hyperparams:
# 1. gradient clipping strength
# 2. option of linear decay action space exploration schedule


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device I am going to use: ", device)

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


# the approach to reformat the joint state x I came up with, leaving here for reference
# def xmem_to_oi(xmem, ith_agent, state_size):
#     """
#         input format of [batch_size, dx], torch tensor
#         output format of [batch_size, ds=dx/N], torch tensor
#     """
#     start = ith_agent * state_size
#     end=start+state_size
#     if xmem.shape[0] == 1:
#         return xmem[start:end].unsqueeze(0)
#     elif xmem.shape[0] > 1:
#         return xmem[:, start:end]
#     else:
#         raise ValueError("xmem has incorrect format")


def xmem_to_oi(xmem, ith_agent, state_size):
    """
        input format of [batch_size, dx], torch tensor
        output format of [batch_size, ds=dx/N], torch tensor
    """
    ith_agent = torch.tensor([ith_agent]).to(device)
    return xmem.reshape(-1, 2, state_size).index_select(1, ith_agent).squeeze(1)


class DDPGAGENT(object):
    """
    DDPG Agent in the context of MADDPG
    """

    def __init__(self, state_size, action_size, n_agents, random_seed=RANDOM_SEED,
                 lr_a=LR_ACTOR, lr_c=LR_CRITIC):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state for each agent
            action_size (int): dimension of each action for each agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # Actor Networks
        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_a, weight_decay=WEIGHT_DECAY_actor)

        # Critic Networks
        self.critic_local = Critic(n_agents * state_size, n_agents * action_size, seed=random_seed).to(device)
        self.critic_target = Critic(n_agents * state_size, n_agents * action_size, seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_c,
                                           weight_decay=WEIGHT_DECAY_critic)

        # Initialize the target and local networks with the same parameters (key for success)
        self.targets_update(tau=0.0) # soft_update(tau=0.0) <=> hard_update

    def noisy_act(self, states, noise_scale):
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += noise_scale * self.add_noise2()
        return np.clip(actions, -1, 1)

    def act(self, states, noise_scale=0.0, add_noise=True):
        if not add_noise:
            if noise_scale>0.0:
                raise TypeError("when add_noise==True, noise_scale must be zero")
            else:
                return self.noisy_act(states, 0.0)
        else:
            return self.noisy_act(states, noise_scale)

    def add_noise2(self):
        # get this method from: https://github.com/gtg162y/DRLND/blob/master/P3_Collab_Compete/Tennis_Udacity_Workspace.ipynb
        noise = 0.5 * np.random.randn(1, self.action_size)  # sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise

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
        xmems, actor_full_actions, actions, rewards, dones, next_xmems, critic_full_next_actions = experiences

        # Critic Learning (Deep Q-learning)
        Q_target_next = self.critic_target(next_xmems, critic_full_next_actions)
        Q_target = rewards + gamma * Q_target_next * (1 - dones)
        Q_expected = self.critic_local(xmems, actions)
        critic_loss = F.mse_loss(input=Q_expected, target=Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Learning (Deterministic Policy Gradients w.r.t Q-function)
        actor_loss = -self.critic_local.forward(xmems, actor_full_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def targets_update(self, tau):
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MADDPGAGENT():
    """
    Implementation MADDPG multi-agent system class
    """
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
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
        self.seed = random.seed(random_seed) # control replay buffer
        self.t_step = 0
        self.agents = dict([(i, None) for i in range(n_agents)])
        for i in range(self.n_agents):
            self.agents[i]=DDPGAGENT(state_size, action_size, n_agents)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise = GaussianExplorationNoise(NOISE_START, NOISE_END, STEPS_BEFORE_LEARNING, EXPLORATION_DECAY)

    def reset(self):
        for i in range(self.n_agents):
            self.agents[i].reset()

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(xenv_to_xmem(state),
                        xenv_to_xmem(action), #todo, debug compare to the working version
                        reward,
                        xenv_to_xmem(next_state),
                        done)

        self.t_step = self.t_step + 1

        if self.t_step % LEARN_EVERY == 0 and self.t_step > STEPS_BEFORE_LEARNING:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_LEARN):
                    for ith_agent in range(self.n_agents):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA, ith_agent)
                    self.multi_agents_targets_update(tau=TAU)

    def multi_agents_act(self, states, add_noise=True):
        """return actions of all agents"""
        if add_noise:
            noise_scale=self.noise.current_noise_scale(self.t_step)
        else:
            noise_scale=0.0

        actions = []
        for agent_id, agent in self.agents.items():
            o_i=xenv_to_oi(states, agent_id)
            action = agent.act(o_i, noise_scale, add_noise)
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def multi_agents_targets_update(self, tau):
        for ith_agent in range(self.n_agents):
            self.agents[ith_agent].targets_update(tau=tau)

    def learn(self, experiences, gamma, ith_agent):
        xmems, full_actions, full_rewards, next_xmems, full_dones = experiences

        critic_full_next_actions = torch.zeros(full_actions.shape, dtype=torch.float, device=device)
        for jth_agent in range(self.n_agents):
            agent_next_state=xmem_to_oi(next_xmems, jth_agent, self.state_size)
            start = jth_agent * self.action_size
            end = start + self.action_size
            critic_full_next_actions[:, start:end] = self.agents[jth_agent].actor_target.forward(agent_next_state)

        agent_state = xmem_to_oi(next_xmems, ith_agent, self.state_size)
        actor_full_actions = full_actions.clone()
        start = ith_agent * self.action_size
        end = start + self.action_size
        actor_full_actions[:, start:end]=self.agents[ith_agent].actor_local.forward(agent_state)

        ddpg_experiences = (xmems, actor_full_actions, full_actions,
                            full_rewards[:, ith_agent].view(-1, 1), full_dones[:, ith_agent].view(-1, 1),  # giving only individual rewards/dones for each agent
                            next_xmems, critic_full_next_actions)
        self.agents[ith_agent].learn(ddpg_experiences, gamma)

    def save_params(self, save_dir='./data/maddpg_test/'):
        for ith_agent, agent in self.agents.items():
            actor_params_path=os.path.join(save_dir,'checkpoint_actor_' + str(ith_agent) + '.pth')
            critic_params_path=os.path.join(save_dir,'checkpoint_critic_' + str(ith_agent) + '.pth')
            torch.save(agent.actor_local.state_dict(), actor_params_path)
            torch.save(agent.critic_local.state_dict(), critic_params_path)

    def load_params(self, load_dir='./data/maddpg_test/', load_all=True):
        # todo, expose load_all, for the future possible option of loading not all agents
        if load_all:
            for ith_agent, agent in self.agents.items():
                actor_params_path=os.path.join(load_dir, 'checkpoint_actor_' + str(ith_agent) + '.pth')
                critic_params_path=os.path.join(load_dir, 'checkpoint_critic_' + str(ith_agent) + '.pth')
                agent.actor_local.load_state_dict(torch.load(actor_params_path))
                agent.critic_local.load_state_dict(torch.load(critic_params_path))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
        the ReplayBuffer here is shared by all DDPG agents in this MADDPG implementation, thus the experience is stored in
        multi-agent manner:
        The experience tuple has the following:
            N: Number of agents

            1. x, joint state of all agents=(s1,...sN)                   format: (N*dS,) np.array
            2. full_action, joint actions of all agents=(a1,...,aN)      format: (N*dA,) np.array
            3. full_reward,                                              format: (N,)    np.array
            4. next_x, joint next state of all agents                    format: (N*dS,) np.array
            5. full_done,                                                format: (N,)    np.array

        The sampled experiences will become torch.Tensor format as:
            Nb: mini-batch size,
            the format of entry in the tuple becomes: (Nb,) + entry.shape
            and will be named when being used : entry_names (e.g. xs)
    """

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["x", "action", "reward", "next_x", "done"])
        self.seed = random.seed(seed)

    def add(self, x, action, reward, next_x, done):
        """Add a new experience to memory."""
        e = self.experience(x, action, reward, next_x, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        xs = torch.from_numpy(np.array([e.x for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_xs = torch.from_numpy(np.array([e.next_x for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (xs, actions, rewards, next_xs, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class GaussianExplorationNoise:
    '''
    Gaussian Exploration Noise for action space exploration:
        noise_scale is exponentially decaying as agents makes more steps (t), precisely
                       1.  e_0                               ,if t<t_0
        noise_scale =  2.  e_0 x alpha^( (t-t_0)//L_factor ) ,if t>t_0 and value of 2.<e_T
                       3.  e_T                               ,if value of 2.>e_T
    '''
    def __init__(self, epsilon_0, epsilon_end, t_0, decay_factor):
        self.e_0=epsilon_0
        self.e_T=epsilon_end
        self.t_0=t_0
        self.alpha=decay_factor

    def current_noise_scale(self, t):
        if t<self.t_0:
            return self.e_0
        else:
            return max( self.e_0*self.alpha**( (t-self.t_0)//L_FACTOR), self.e_T)