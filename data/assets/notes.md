- Technical Details:

1. Naming:
    - N agents in the system
    - Nb mini-batch size for optimization
    - x_env: joint states from the environment: (N, dS)
    - x_mem: joint states from the memory: (N*dS,) or sampled x_mem (Nb,N*dS)
    - o_i:   individual state for ith_agent: (dS,) 

2. Initialize all A/C in DDPG within the MADDPG system
    - making sure that target and local of all A/Cs must be initialized with the same values!

3. in MADDPGAGENT.learn():     
    - critic_full_next_action =[mu_1(x1'_mem), mu_2(x2'_mem), mu_3(x3'_mem)..., mu_{n_agents}(x_{n_agents}_mem)]
    - actor_full_actions = a\ai-U-{mu_i(o_i)}

4. PyTorch autograd is very elegant for both A/C learning in DDPG, check DDPGAGENT.learn
5. There are 3 sources of randomness in this project: NN, Buffer, Noise generation    

To Work On:
1. FCNetwork is not directly applicable into this program, need some time to figure out why
2. Add features of PyToch 1. ModelViz, 2. TensorBoard; 3. AWS interface for efficient debugging/testing
