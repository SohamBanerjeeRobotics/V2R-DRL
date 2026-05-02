import torch
import numpy as np
import time
import pybullet as p

from Training_env_1 import TrainingEnvOne
from ADDPG_Training_script import Agent   # reuse same class

env = TrainingEnvOne(render=True)

agent = Agent(14, 2)
agent.load("addpg_final.pth")

obs, _ = env.reset()

# 🔥 CUSTOM TARGET
#target = np.array([3.0, 3.0])
#env.target = target
#p.loadURDF("sphere2.urdf", [target[0], target[1], 0.2])

for _ in range(50000):

    s = torch.FloatTensor(obs).unsqueeze(0)

    with torch.no_grad():
        action = agent.actor(s).squeeze(0).numpy()

    obs, _, term, trunc, _ = env.step(action)

    time.sleep(0.02)

    if term or trunc:
        obs, _ = env.reset()

        # reapply target
        #env.target = target
        #p.loadURDF("sphere2.urdf", [target[0], target[1], 0.2])
