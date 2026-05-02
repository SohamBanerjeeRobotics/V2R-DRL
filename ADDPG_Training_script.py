import numpy as np
import random
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from Training_env_1 import TrainingEnvOne

# =========================================================
# GLOBAL CONTROL
# =========================================================
global_step = 0
step_lock = threading.Lock()
max_steps = 20_000_000

# =========================================================
# REPLAY BUFFER (THREAD SAFE)
# =========================================================
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()

    def add(self, transition):
        with self.lock:
            self.buffer.append(transition)

    def sample(self, batch_size):
        with self.lock:
            batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)

    def __len__(self):
        with self.lock:
            return len(self.buffer)

# =========================================================
# NETWORKS
# =========================================================
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, a_dim)
        )

    def forward(self, x):
        out = self.net(x)
        lin = torch.sigmoid(out[:, 0:1])
        ang = torch.tanh(out[:, 1:2])
        return torch.cat([lin, ang], dim=1)


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

# =========================================================
# AGENT
# =========================================================
class Agent:
    def __init__(self, s_dim, a_dim):
        self.actor = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)

        self.target_actor = Actor(s_dim, a_dim)
        self.target_critic = Critic(s_dim, a_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.tau = 0.005

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

# =========================================================
# ENV FACTORY
# =========================================================
def make_env():
    return TrainingEnvOne(render=False)


class OUNoise:
    def __init__(self, dim, theta=0.15, sigma=0.2):
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(dim)

    def sample(self):
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

    def reset(self):
        self.state = np.zeros_like(self.state)

# =========================================================
# COLLECTOR (FIXED)
# =========================================================
def collector(env_fn, agent, buffer, tid, stop_event):
    global global_step

    env = env_fn()
    state, _ = env.reset()

    ep_reward, ep_steps = 0, 0
    
    noise = OUNoise(2)
    while not stop_event.is_set():

        with step_lock:
            if global_step >= max_steps:
                stop_event.set()
                break

        s_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = agent.actor(s_tensor).squeeze(0).numpy()

        action = action + noise.sample()
        action = np.clip(action, [0.0, -1.0], [1.0, 1.0])

        # 🔥 FIX: protect against env crash
        try:
            s2, r, term, trunc, _ = env.step(action)
        except Exception as e:
            print(f"[Collector {tid}] Env crash: {e}")
            state, _ = env.reset()
            ep_reward, ep_steps = 0, 0
            noise.reset()
            continue

        done = term or trunc

        buffer.add((state, action, r, s2, done))

        state = s2
        ep_reward += r
        ep_steps += 1

        with step_lock:
            global_step += 1

        if done:
            print(f"[Collector {tid}] Reward: {ep_reward:.2f} | Steps: {ep_steps} | Global: {global_step}")
            state, _ = env.reset()
            ep_reward, ep_steps = 0, 0
            noise.reset()

# =========================================================
# TRAINER
# =========================================================
def trainer(agent, buffer, stop_event):
    batch_size = 64
    step = 0

    while not stop_event.is_set():

        if len(buffer) < batch_size:
            time.sleep(0.01)
            continue

        s, a, r, s2, d = buffer.sample(batch_size)

        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d).unsqueeze(1)

        with torch.no_grad():
            next_a = agent.target_actor(s2)
            q_target = agent.target_critic(s2, next_a)
            y = r + agent.gamma * (1 - d) * q_target

        q = agent.critic(s, a)
        critic_loss = F.mse_loss(q, y)

        agent.critic_opt.zero_grad()
        critic_loss.backward()
        agent.critic_opt.step()

        actor_loss = -agent.critic(s, agent.actor(s)).mean()

        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        # soft update
        for p, tp in zip(agent.actor.parameters(), agent.target_actor.parameters()):
            tp.data.copy_(agent.tau * p.data + (1 - agent.tau) * tp.data)

        for p, tp in zip(agent.critic.parameters(), agent.target_critic.parameters()):
            tp.data.copy_(agent.tau * p.data + (1 - agent.tau) * tp.data)

        step += 1

        if step % 100 == 0:
            print(f"[TRAIN] Step {step} | Buffer {len(buffer)} | Loss {critic_loss.item():.4f}")

        if step % 5000 == 0:
            agent.save("addpg_model.pth")
            print("Model saved")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    agent = Agent(14, 2)
    buffer = ReplayBuffer()
    stop_event = threading.Event()

    collectors = []
    for i in range(4):
        t = threading.Thread(target=collector, args=(make_env, agent, buffer, i, stop_event))
        t.start()
        collectors.append(t)

    t_train = threading.Thread(target=trainer, args=(agent, buffer, stop_event))
    t_train.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in collectors:
        t.join()

    t_train.join()

    agent.save("addpg_final.pth")
    print("Training complete")
