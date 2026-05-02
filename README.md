# V2R-DRL
Continuous Control of Mobile Robots for Mapless Navigation 

1. This repo contains two training Environements (Training_env_1.py and Training_env_2.py) and one test environment (Test_env_1.py)
2. To visualize the envs run env_visualize.py with adjustments.
3. ADDPG_Training_script.py runs the Asynchronous Deep-Deterministic Policy Gradient algorithm for training the policy.
4. The policy is saved in addpg_final.pth file
5. Test_policy.py helps in checking the policy.


For furthur reference on reqard design and other infos regarding ADDPG or RL in general or its application in this specific domain, go through the 
following paper: Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation(https://arxiv.org/abs/1703.00420)
