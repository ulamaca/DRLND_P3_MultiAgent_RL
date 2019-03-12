from unityagents import UnityEnvironment
import environment
from collections import deque
import numpy as np
from maddpg_agent_v0 import MADDPGAGENT
import os

n_epsisodes=2000
training=True
seed = 10
print_every = 100

if __name__ == "__main__":
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


    agent = MADDPGAGENT(state_size=state_size, action_size=action_size, random_seed=seed, n_agents=num_agents)
    for i in range(n_epsisodes):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=training)[brain_name]  # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores_deque = deque(maxlen=print_every)
        scores_history = []
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        agent.reset() # reset action exploration noise # check for noise adding
        while True:
            actions = agent.multi_agents_act(states)           # select an action (for each agent)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards       # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            agent.step(states, actions, rewards, next_states, dones) # must have a bug!! todo, check learning rules for actor
            if np.any(dones):                                  # exit loop if episode finished
                break

        episdoe_score=np.max(scores)
        scores_history.append(episdoe_score)
        scores_deque.append(episdoe_score)
        mean_score_deque=np.mean(scores_deque)
        print('\rScore (max over agents) from episode {}: {:.4f}'.format(i, episdoe_score), end="")
        #print('\rAll Scores from episode {}: {:.4f}, {:.4f}'.format(i, scores[0], scores[1]))
        if i>0 and i % print_every==0:
            print('\rMean Score (max over agents) from episode {}: {:.4f}'.format(i, mean_score_deque))

        if mean_score_deque > 0.5:
            print("Problem Solved!")
            break

    # save agent params:
    #torch.save(agent.state_dict(), './data/ppo_gae/checkpoint.pth')
    # scores can be plotted
    with open(os.path.join('./data/maddpg_test/progress.txt'), 'w') as myfile:
        myfile.write(str(scores_history))
    myfile.close()

