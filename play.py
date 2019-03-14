import environment
from collections import deque
import numpy as np
from maddpg_agent import MADDPGAGENT
import time
import os
import argparse

n_epsisodes=5000
training=True
seed = 10
print_every = 100
state_size=24
action_size=2
num_agents=2
EXPS_ROOT_PATH = './data'

parser=argparse.ArgumentParser(description="Play a trained MADDPG system in Unity Tennis Environment")
parser.add_argument('-n', '--name', type=str, metavar='', default='no-name-exp', help="name of the training run (default no-name-exp)")
args=parser.parse_args()


if __name__ == "__main__":
    agent = MADDPGAGENT(state_size=state_size, action_size=action_size, random_seed=seed, n_agents=num_agents)
    exp_dir = os.path.join(EXPS_ROOT_PATH, args.name)
    agent.load_params(load_dir=exp_dir)

    env = environment.UnityMLVectorMultiAgent(evaluation_only=True)
    states = env.reset()
    scores = np.zeros(num_agents)
    steps=0
    while True:
        actions = agent.multi_agents_act(states, add_noise=False)
        next_states, rewards, dones = env.step(actions)
        scores+=rewards
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
        steps+=1
        if np.any(dones):                                  # exit loop if episode finished
            break

    final_score=np.max(scores)
    print('\rWe are team MADDPG, we got {:.4f}(max over two of us) in {} steps'.format(final_score, steps), end="")



