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
2. Add features of PyToch 
    1. ModelViz, 
    2. TensorBoard; 
    3. AWS interface for efficient debugging/testing
        * 03.20 I claimed my AWS credits for the Nano-degree
        * 03.22 Try to construct a AWS AMI instance but failed to connect to it because of the key pair 
            * till the point of 6. in the Udacity reference:
            * the error message:
                gj@gj-XPS-13-9360:~/Downloads$ ssh -i play_drl.pem ubuntu@13.112.104.7          
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            Permissions 0664 for 'play_drl.pem' are too open.
            It is required that your private key files are NOT accessible by others.
            This private key will be ignored.
            Load key "play_drl.pem": bad permissions
            * Problem resolved by execute: chmod 400 ~/path_to_pem_file
                * cf: https://stackoverflow.com/questions/9270734/ssh-permissions-are-too-open-error
            * log:
                - 3.28.2019 play MADDPG on AWS instances
                    - challenge: no video rendering on AWS for runing RL training procedure
                        - naive solution: use jupyter notebook to run
                    - git clone my github (http)
                    - sudo pip3 install requirements.txt
                    - download linux Tennis env: wget <web address>
                    - create a screen env. for run
                        - python3 run.py 
                        - leave the process through ctrl.+a d => use screen -ls to find the process and screen -r to resume

3. Merge the three projects in the nano-degree 
4. Hyperparam selection platform:
    * https://www.comet.ml/
    * https://www.floydhub.com/
    * package: https://ray.readthedocs.io/en/latest/tune.html