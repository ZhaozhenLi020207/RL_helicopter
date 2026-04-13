"""
SAC 直升机牵引入库训练
【CUDA 全速版】
环境文件保持不变：helicopter_env.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime

# 从你原来的环境导入（完全不动你的环境）
from helicopter_env import HelicopterInboundKinematicsEnv

# ======================== CUDA 全速加速 ========================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(16)

# ======================== 网络结构 ========================
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = action * self.action_scale + self.action_bias
        return action, log_prob

# ======================== 极速经验池 ========================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.s[self.pos] = state
        self.a[self.pos] = action
        self.r[self.pos] = reward
        self.s2[self.pos] = next_state
        self.d[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.tensor(self.s[idx], device=self.device),
            torch.tensor(self.a[idx], device=self.device),
            torch.tensor(self.r[idx], device=self.device),
            torch.tensor(self.s2[idx], device=self.device),
            torch.tensor(self.d[idx], device=self.device)
        )

# ======================== SAC Agent ========================
class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005

        # 动作范围（完全和你环境一致）
        self.a_low = np.array([-1.0, 0.005], dtype=np.float32)
        self.a_high = np.array([1.0, 0.03], dtype=np.float32)
        self.scale = torch.tensor((self.a_high - self.a_low) / 2, device=device)
        self.bias = torch.tensor((self.a_high + self.a_low) / 2, device=device)

        # 网络
        self.actor = GaussianPolicy(state_dim, action_dim, 128, self.scale, self.bias).to(device)
        self.q1 = SoftQNetwork(state_dim, action_dim, 128).to(device)
        self.q2 = SoftQNetwork(state_dim, action_dim, 128).to(device)
        self.tq1 = SoftQNetwork(state_dim, action_dim, 128).to(device)
        self.tq2 = SoftQNetwork(state_dim, action_dim, 128).to(device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        # 优化器
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=1e-3)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=1e-3)

        # 自适应温度
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.opt_alpha = optim.Adam([self.log_alpha], lr=1e-3)

        # 经验池
        self.buffer = ReplayBuffer(200000, state_dim, action_dim, device)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            a, _ = self.actor.sample(s, deterministic)
        return a.cpu().numpy()[0]

    def update(self, batch_size):
        if self.buffer.size < batch_size:
            return

        s, a, r, s2, done = self.buffer.sample(batch_size)

        # 更新 Critic
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            tq = torch.min(self.tq1(s2, a2), self.tq2(s2, a2)) - self.alpha * logp2
            target_q = r + (1 - done) * self.gamma * tq

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)

        self.opt_q1.zero_grad()
        loss_q1.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        loss_q2.backward()
        self.opt_q2.step()

        # 更新 Actor
        a_new, logp = self.actor.sample(s)
        q_min = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        loss_actor = (self.alpha * logp - q_min).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # 更新 Alpha
        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()
        self.alpha = self.log_alpha.exp()

        # 软更新
        for t, p in zip(self.tq1.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
        for t, p in zip(self.tq2.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

# ======================== 训练主程序 ========================
def train():
    env = HelicopterInboundKinematicsEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(state_dim, action_dim, device)

    max_episodes = 1500
    batch_size = 512
    update_after = 2000
    updates_per_step = 4
    save_dir = f"SAC_Heli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    success_window = deque(maxlen=50)
    best_reward = -np.inf
    start_time = time.time()

    print("===== SAC 训练开始（RTX A5000 全速）=====")

    for episode in range(1, max_episodes + 1):
        s, _ = env.reset()
        total_r = 0
        steps = 0
        term, trunc = False, False

        while not (term or trunc):
            # 探索策略
            det = episode < 80
            a = agent.select_action(s, det)

            # 环境步进（完全使用你的环境）
            s2, r, term, trunc, _ = env.step(a)
            agent.buffer.add(s, a, r, s2, term or trunc)

            s = s2
            total_r += r
            steps += 1

            # 网络更新
            if agent.buffer.size > update_after:
                for _ in range(updates_per_step):
                    agent.update(batch_size)

        # 记录
        episode_rewards.append(total_r)
        e_fm = env.x_fm - env.track_centerline(env.y_fm)
        theta_rel = env.theta - env.track_angle(env.y_fm)
        success = term and abs(e_fm) < 0.1 and abs(theta_rel) < np.deg2rad(12)
        success_window.append(1 if success else 0)
        sr = np.mean(success_window)

        # 保存最佳模型
        if total_r > best_reward:
            best_reward = total_r
            torch.save(agent.actor.state_dict(), f"{save_dir}/best_actor.pth")

        # 日志
        if episode % 10 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            elapsed = (time.time() - start_time) / 60
            print(f"Ep {episode:4d} | R:{total_r:6.1f} | avgR:{avg_r:5.1f} | SR:{sr:.1%} | Time:{elapsed:.1f}min")

    print(f"\n训练完成！耗时：{(time.time() - start_time)/60:.1f} min")

if __name__ == "__main__":
    train()