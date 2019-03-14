[image1]: ./data/noise_process.png
[image2]: ./data/alg_maddpg.png
[image3]: ./data/maddpg_report_result.png
[anime1]: ./data/maddpg_demo.gif

### **Algorithms**
In this project, I applied MADDPG (Multi-Agent Deep Deterministic Policy Gradients) to solve a 2-agent Tennis game (Unity Tennis environment). 
DDPG is one of the most basic forms of deep reinforcement learning agent for solving continuous control problems. By using deterministic policy, policy gradients with respect to 
the value function is computable. Therefore, DDPG algorithm replaces such policy in DQN and re-formulate it as an actor-critic style learning algorithms where 
Q-function is learned using deep-Q learning and policy learning part uses the aforementioned gradients. 

In MADDPG paper, the authors develop some tricks to apply DDPG in multi-agent setting. The most notorious problem in multi-agent control problem is that
the environment is not a MDP anymore if any policy in the system is changed. Such problem will lead to great confusion (e.g. contradictory gradients for competing agents) during learning.
One major trick here is that if the actions of all agents are given, then it will be a MDP again. 
To utilize this property, one can represent centralized value functions that takes into account all states/actions from all agents.
Then, one can train a multi-agent system with all agents having a centralized critic and (local) policy. Since MADDPG does not assume specific multi-agent setting, it has been shown
to perform well in competitive, cooperative and mixed scenarios. 
   


### **Implementation**
My implementation is based on the idea of Algorithm in Appendix in MADDPG paper: ![Algorithm 1][image2]



Note that my noise process is listed in the Appendix section *2. All agents share a single replay buffer but samples their own experiences separately during learning.
For each learning step, each agent will be updated in sequence. 


### **Results**  

#### **Statistics**

![Figure1][image3]
The agent solves the environment in 1306 episodes. The total time elapsed is 5787.50492143631 seconds (~100 minutes in my Dell XPS13 laptop, 4CPU, 16G memory)

**Video recording of a trained agent**
![trained agent][anime1]

### **Future Work**
- Extend our DDPG implementation to SOTA (e.g. TD3) to see potential improvement 
- Using hindsight experience replay (HER) or parameter noise to boot data efficiency/ exploration.
- Add tensorboard and develop AWS functions to get better debugging experiences. 
- For environment having great symmetry like Tennis, it is worthy to try self-play training to see if it works. PPO in project 2 could be a good start.
- Work on solving the harder Soccer environment. 


### **Reference**
Research Papers:
- [MADDPG 2017](https://arxiv.org/pdf/1706.02275)
- [DDPG 2015](https://arxiv.org/abs/1509.02971)

Related works:
- [drlnd_collaboration_and_competition(by Daniel Barbosa)](https://github.com/danielnbarbosa/drlnd_collaboration_and_competition) 
- [DRLND/P3_Collab_Compete(by Amit Patel)](https://github.com/gtg162y/DRLND/tree/master/P3_Collab_Compete)
    - Thanks for the works/authors, really helpful for my implementation



### **Appendix** 
#### Hyperparamerters:
| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay Buffer Size                  | 5e4   |
| Mini-Batch Size                     | 200   |
| LEARN EVERY ? Steps                 | 2     |
| #Updates per LEARNING STEP          | 4     |
| #Steps BEFORE LEARNING (t_0)             | 5500 (~300 episodes) |
| Gamma (discount factor)             | 0.99  |
| Agents' Model Type                  | MLP  |
| Agents' Model Arch (Policy)           | [in, 256, 256, out] |
| Agents' Model Arch (Critic)*1         | [x, 256, 256, 128, 1] |
|                                       | [    a              ] |
| Actor Learning Rate (All agents)      | 1e-4 |
| Critic Learning Rate (All agents)     | 5e-4 |
| Noise Start (action exploration*2)      | 1.0  |
| Noise End   (action exploration)      | 0.1  |
| Decay Rate   (action exploration)     | 0.999 |
| L_Factor    (action exploration)      | 80 |
| Seed (Control Randomness of Buffer, NNs, Action Noise) | 70 |


- *1. The critic first process the state x with a layer of NN and then use concatenation of its outputs and actions as the first hidden layer for futher forward pass.
- *2. The action exploration noise is generated through N(0,\epsilon(t)) where \episilon(t) is the following:
![Equation1][image1]
- *3. No weight decay for any A/C in the MADDPG system is used in this implementation