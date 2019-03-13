import environment
from collections import deque
import numpy as np
from maddpg_agent import MADDPGAGENT
import time
import os

n_epsisodes=5000
training=True
seed = 10
print_every = 100
state_size=24
action_size=2
num_agents=2


# def seeding(seed=10):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


if __name__ == "__main__":
    agent = MADDPGAGENT(state_size=state_size, action_size=action_size, random_seed=seed, n_agents=num_agents)
    env=environment.UnityMLVectorMultiAgent()

    scores_history = []
    scores_deque = deque(maxlen=print_every)
    t0=time.time()
    for i in range(n_epsisodes):
        states = env.reset()
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        #agent.reset()                                          # reset all noise mu_i
        while True:
            actions = agent.multi_agents_act(states)           # select an action (for each agent)
            next_states, rewards, dones = env.step(actions)
            scores+=rewards
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):                                  # exit loop if episode finished
                break

        episdoe_score=np.max(scores)
        scores_history.append(episdoe_score)
        scores_deque.append(episdoe_score)
        mean_score_deque=np.mean(scores_deque)
        print('\rScore (max over agents) from episode {}: {:.4f}'.format(i, episdoe_score), end="")
        if i>0 and i % print_every==0:
            print('\rMean Score (max over agents) from episode {}: {:.4f}'.format(i, mean_score_deque))
        if mean_score_deque > 0.5:
            print("\nProblem Solved!")
            break
    t1=time.time()

    print("\nTotal time elapsed: {} seconds".format(t1-t0))
    # save agent params:
    agent.save_params()
    with open(os.path.join('./data/maddpg_test/progress.txt'), 'w') as myfile:
        myfile.write(str(scores_history))
    myfile.close()

