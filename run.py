import environment
from collections import deque
import numpy as np
from maddpg_agent import MADDPGAGENT
import time
import os
import argparse


training=True
seed = 10
print_every = 100
state_size=24
action_size=2
num_agents=2
EXPS_ROOT_PATH = './data'

parser=argparse.ArgumentParser(description="train a MADDPG system in Unity Tennis Environment")
parser.add_argument('-n', '--name', type=str, metavar='', default='no-name-exp', help="name of the training run (default no-name-exp)")
parser.add_argument('-ne', '--num_episodes', type=int, metavar='', default=3000, help="")
args=parser.parse_args()


if __name__ == "__main__":
    agent = MADDPGAGENT(state_size=state_size, action_size=action_size, random_seed=seed, n_agents=num_agents)
    env=environment.UnityMLVectorMultiAgent()

    scores_history = []
    scores_deque = deque(maxlen=print_every)
    t0=time.time()
    for i in range(args.num_episodes):
        states = env.reset()
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
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
        print('\rScore (max over agents) from episode {}: {:.4f}, current noise scale {}'.format(i, episdoe_score,
                                                                                                 agent.noise.current_noise_scale(agent.t_step)), end="")
        if i>0 and i % print_every==0:
            print('\nMean Score (max over agents) from episode {}: {:.4f}'.format(i, mean_score_deque))
        if mean_score_deque > 0.5:
            print("\nProblem Solved!")
            break
    t1=time.time()

    print("\nTotal time elapsed: {} seconds".format(t1-t0))

    # save agent params:
    exp_dir = os.path.join(EXPS_ROOT_PATH, args.name)
    os.makedirs(exp_dir, exist_ok=True)
    agent.save_params(save_dir=exp_dir)
    with open(os.path.join(exp_dir, 'progress.txt'), 'w') as myfile:
        myfile.write(str(scores_history))
    myfile.close()

