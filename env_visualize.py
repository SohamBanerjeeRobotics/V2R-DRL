from Training_env_1 import TrainingEnvOne
import time
env = TrainingEnvOne(render=True)

obs, _ = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(reward, terminated)
    time.sleep(0.02)

    if terminated or truncated:
        obs, _ = env.reset()
