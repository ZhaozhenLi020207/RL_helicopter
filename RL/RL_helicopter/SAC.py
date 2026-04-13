"""
基于SAC (Soft Actor-Critic)算法的直升机自动牵引入库训练 - 高速版
运动方向：从降落区域（y=3.55）向机库（y=0）运动
【CUDA 全速优化版 + 环境加速】
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')  # 禁用matplotlib交互
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime
from helicopter_env import HelicopterInboundKinematicsEnv  # 使用快速版环境

# ========== CUDA 全局加速 ==========
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# 设置环境变量加速
os.environ['OMP_NUM_THREADS'] = '1'  # 避免numpy多线程开销
os.environ['MKL_NUM_THREADS'] = '1'

# ========== 网络定义 ==========
class SoftQNetwork(nn.Module):
    """Soft Q网络（Critic）"""
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
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """高斯策略网络（Actor）"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
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

    def evaluate(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        action_normalized = (action - self.action_bias) / self.action_scale
        action_normalized = torch.clamp(action_normalized, -0.999, 0.999)
        z = torch.atanh(action_normalized)

        normal = Normal(mean, std)
        log_prob = normal.log_prob(z) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob


# ========== Replay Buffer ==========
class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[indices], device=self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions[indices], device=self.device, dtype=torch.float32)
        rewards = torch.tensor(self.rewards[indices], device=self.device, dtype=torch.float32)
        next_states = torch.tensor(self.next_states[indices], device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.dones[indices], device=self.device, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


# ========== SAC Agent ==========
class SACAgent:
    """Soft Actor-Critic Agent"""
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device(config['device'])

        self.action_low = np.array([-1.0, 0.005], dtype=np.float32)
        self.action_high = np.array([1.0, 0.03], dtype=np.float32)
        self.action_scale = torch.tensor((self.action_high - self.action_low) / 2, device=self.device)
        self.action_bias = torch.tensor((self.action_high + self.action_low) / 2, device=self.device)

        self.actor = GaussianPolicy(
            state_dim, action_dim,
            config['hidden_dim'],
            self.action_scale, self.action_bias
        ).to(self.device)

        self.q1 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config['critic_lr'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config['critic_lr'])

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['alpha_lr'])

        self.buffer = ReplayBuffer(
            config['buffer_capacity'],
            state_dim,
            action_dim,
            self.device
        )

        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.success_history = []

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic)
        action_np = action.cpu().numpy()[0]

        if len(action_np) < 2:
            action_np = np.concatenate([action_np, np.array([0.01])])[:2]

        if not deterministic and np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, size=2)
            action_np = np.clip(action_np + noise, self.action_low, self.action_high)

        return action_np.astype(np.float32)

    def update(self):
        if len(self.buffer) < self.config['batch_size']:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config['batch_size'])

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            t_q1 = self.target_q1(next_states, next_actions)
            t_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(t_q1, t_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config['gamma'] * (1 - dones) * target_q

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        tau = self.config['tau']
        for t, p in zip(self.target_q1.parameters(), self.q1.parameters()):
            t.data.copy_(tau * p.data + (1 - tau) * t.data)
        for t, p in zip(self.target_q2.parameters(), self.q2.parameters()):
            t.data.copy_(tau * p.data + (1 - tau) * t.data)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load_model(self, path):
        cp = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(cp['actor'])
        self.q1.load_state_dict(cp['q1'])
        self.q2.load_state_dict(cp['q2'])
        self.target_q1.load_state_dict(cp['target_q1'])
        self.target_q2.load_state_dict(cp['target_q2'])
        self.log_alpha = cp['log_alpha']
        self.alpha = self.log_alpha.exp()


# ========== 训练环境包装器（简化版，减少开销） ==========
class TrainingWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.last_action = np.zeros(2)
        self.last_e_fm = None

    def step(self, action):
        if len(action) < 2:
            action = np.concatenate([action, np.array([0.01])])[:2]
        obs, reward, terminated, truncated, info = self.env.step(action)
        e_fm, theta_rel, e_p, tail_angle, y_remaining = obs

        # 简化的额外奖励（保持原逻辑）
        action_change = np.abs(action - self.last_action).sum()
        if action_change < 0.05:
            reward += 0.01

        if self.last_e_fm is not None:
            improvement = abs(self.last_e_fm) - abs(e_fm)
            if improvement > 0:
                reward += 0.5 * improvement

        self.last_e_fm = e_fm
        self.last_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(2)
        self.last_e_fm = obs[0]
        return obs, info


# ========== 训练可视化（简化版，减少绘图开销） ==========
class TrainingVisualizer:
    def __init__(self, save_dir='training_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.rolling_success = deque(maxlen=50)

    def update(self, episode, reward, length, success):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.rolling_success.append(1 if success else 0)
        self.success_rates.append(np.mean(self.rolling_success))

        # 减少绘图频率：每50个episode才绘图
        if episode % 50 == 0:
            self._plot(episode)

    def _plot(self, episode):
        # 使用非交互式绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.episode_rewards, alpha=0.5)
        axes[0].set_title('Reward')
        axes[0].grid(True)

        axes[1].plot(self.episode_lengths)
        axes[1].set_title('Length')
        axes[1].grid(True)

        axes[2].plot(self.success_rates)
        axes[2].set_title('Success Rate')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png', dpi=100)
        plt.close(fig)

    def save_final_plot(self):
        # 最终绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.episode_rewards)
        axes[0].set_title('Reward')
        axes[0].grid(True)

        axes[1].plot(self.episode_lengths)
        axes[1].set_title('Length')
        axes[1].grid(True)

        axes[2].plot(self.success_rates)
        axes[2].set_title('Success Rate')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/final_plot.png', dpi=150)
        plt.close(fig)


# ========== 评估函数 ==========
def evaluate_agent(agent, env, num_episodes=10, render=False):
    success_count = 0
    total_rewards = []
    e_fm_limit = env.env.e_fm_limit
    theta_limit = env.env.theta_limit

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_r = 0
        steps = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.select_action(state, deterministic=True)
            s, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            steps +=1
            state = s

        y_fm = env.env.y_fm
        e_fm = env.env.x_fm - env.env.track_centerline(y_fm)
        theta_rel = env.env.theta - env.env.track_angle(y_fm)
        success = terminated and y_fm <= env.env.y_end and abs(e_fm) < e_fm_limit and abs(theta_rel) < theta_limit

        if success: success_count +=1
        total_rewards.append(ep_r)

    avg_r = np.mean(total_rewards)
    sr = success_count / num_episodes
    print(f"评估完成 | 平均奖励: {avg_r:.1f} | 成功率: {sr:.1%}")
    return {'success_rate': sr, 'avg_reward': avg_r}


# ========== 主训练 ==========
def train_sac(config):
    # 使用快速版环境，禁用所有渲染
    base_env = HelicopterInboundKinematicsEnv(render_mode=None, fast_mode=True)
    env = TrainingWrapper(base_env, config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("="*60)
    print("SAC 直升机牵引入库训练 - 超高速版")
    print(f"状态维度 {state_dim} | 动作维度 {action_dim}")
    print(f"设备: {config['device']}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    agent = SACAgent(state_dim, action_dim, config)
    viz = TrainingVisualizer(config['save_dir'])
    start = time.time()
    best_r = -np.inf
    success_count = 0
    e_fm_limit = env.env.e_fm_limit
    theta_limit = env.env.theta_limit

    # 性能计时
    step_times = deque(maxlen=1000)

    for episode in range(1, config['max_episodes']+1):
        state, _ = env.reset()
        ep_r = 0
        ep_len = 0
        terminated, truncated = False, False
        episode_start = time.time()

        while not (terminated or truncated):
            step_start = time.time()

            # 前100个episode探索，之后利用
            det = episode < 100
            action = agent.select_action(state, deterministic=det)
            ns, r, terminated, truncated, _ = env.step(action)
            agent.buffer.push(state, action, r, ns, terminated or truncated)
            state = ns
            ep_r += r
            ep_len += 1

            if len(agent.buffer) > config['learning_starts']:
                for _ in range(config['updates_per_step']):
                    agent.update()

            step_times.append(time.time() - step_start)

        agent.episode_rewards.append(ep_r)
        agent.episode_lengths.append(ep_len)

        y_fm = env.env.y_fm
        e_fm = env.env.x_fm - env.env.track_centerline(y_fm)
        theta_rel = env.env.theta - env.env.track_angle(y_fm)
        success = terminated and y_fm <= env.env.y_end and abs(e_fm) < e_fm_limit and abs(theta_rel) < theta_limit
        if success: success_count += 1

        viz.update(episode, ep_r, ep_len, success)

        if ep_r > best_r:
            best_r = ep_r
            agent.save_model(f"{config['save_dir']}/best_model.pt")

        episode_time = time.time() - episode_start

        if episode % config['log_interval'] == 0:
            avg_r = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards)>=100 else np.mean(agent.episode_rewards)
            sr = success_count / episode
            elapsed = (time.time()-start)/60
            avg_step_time = np.mean(step_times) * 1000 if step_times else 0
            steps_per_sec = 1.0 / np.mean(step_times) if step_times else 0

            print(f"Ep {episode:4d} | R: {ep_r:8.1f} | AvgR: {avg_r:7.1f} | SR: {sr:6.1%} | "
                  f"Steps/s: {steps_per_sec:5.1f} | Time: {elapsed:5.1f}min | EpTime: {episode_time:4.1f}s")

        if episode % config['eval_interval'] == 0:
            print("\n--- 评估中 ---")
            evaluate_agent(agent, env, 20)
            print("--- 评估结束 ---\n")

    print(f"\n训练完成！耗时: {(time.time()-start)/60:.1f} min")
    agent.save_model(f"{config['save_dir']}/final_model.pt")
    viz.save_final_plot()

    final_eval = evaluate_agent(agent, env, 50)
    return agent, final_eval


# ========== 主程序 ==========
if __name__ == "__main__":
    config = {
        'gamma': 0.99,
        'tau': 0.005,
        'alpha_lr': 1e-3,
        'hidden_dim': 128,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,

        'max_episodes': 1500,
        'batch_size': 512,
        'buffer_capacity': 200000,
        'learning_starts': 2000,
        'updates_per_step': 4,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': f'SAC_Heli_Fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 10,
        'eval_interval': 200,
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    import json
    with open(f"{config['save_dir']}/config.json",'w') as f:
        json.dump({k:str(v) if k=='device' else v for k,v in config.items()}, f, indent=4)

    # 打印配置
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    try:
        agent, res = train_sac(config)
        print("\n" + "="*60)
        print(f"训练全部完成！最终成功率: {res['success_rate']:.1%}")
        print("="*60)
    except KeyboardInterrupt:
        print("\n手动停止")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()