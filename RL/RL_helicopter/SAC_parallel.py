"""
基于SAC算法的直升机自动牵引入库训练 - 并行环境版
使用多个并行环境同时收集数据，大幅提升GPU利用率
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime
from helicopter_env import HelicopterInboundKinematicsEnv
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# ========== CUDA 优化 ==========
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'


# ========== 网络定义（与之前相同） ==========
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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


# ========== 并行环境管理器 ==========
class ParallelEnvManager:
    """管理多个并行环境，支持异步step"""

    def __init__(self, num_envs, easy_mode=False):
        self.num_envs = num_envs
        self.envs = []
        self.states = []
        self.dones = []

        # 创建多个独立环境
        for _ in range(num_envs):
            env = HelicopterInboundKinematicsEnv(render_mode=None, fast_mode=True, easy_mode=easy_mode)
            self.envs.append(env)

        # 重置所有环境
        self.reset_all()

    def reset_all(self):
        """重置所有环境"""
        self.states = []
        self.dones = [False] * self.num_envs
        for env in self.envs:
            state, _ = env.reset()
            self.states.append(state)
        return np.array(self.states, dtype=np.float32)

    def step_parallel(self, actions):
        """并行执行所有环境的step（使用列表推导，Python会并行化）"""
        next_states = []
        rewards = []
        terminateds = []
        truncateds = []

        for i, env in enumerate(self.envs):
            if self.dones[i]:
                # 如果已完成，重置环境
                state, _ = env.reset()
                self.states[i] = state
                self.dones[i] = False
                next_states.append(state)
                rewards.append(0.0)
                terminateds.append(False)
                truncateds.append(False)
            else:
                # 执行step
                next_state, reward, terminated, truncated, _ = env.step(actions[i])
                next_states.append(next_state)
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)

                if terminated or truncated:
                    self.dones[i] = True

        self.states = next_states
        return (np.array(next_states, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(terminateds, dtype=bool),
                np.array(truncateds, dtype=bool))


# ========== Replay Buffer（支持批量添加） ==========
class ReplayBuffer:
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

    def push_batch(self, states, actions, rewards, next_states, dones):
        """批量添加经验"""
        batch_size = len(states)
        for i in range(batch_size):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

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

    def select_actions_batch(self, states, deterministic=False):
        """批量选择动作"""
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self.actor.sample(states_tensor, deterministic)
        return actions.cpu().numpy()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic)
        return action.cpu().numpy()[0]

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
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
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

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }

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


# ========== 训练可视化 ==========
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

        if episode % 50 == 0:
            self._plot(episode)

    def _plot(self, episode):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.episode_rewards[-500:], alpha=0.5)
        axes[0].set_title('Reward (last 500)')
        axes[0].grid(True)

        axes[1].plot(self.episode_lengths[-500:])
        axes[1].set_title('Length (last 500)')
        axes[1].grid(True)

        axes[2].plot(self.success_rates)
        axes[2].set_title('Success Rate')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png', dpi=100)
        plt.close(fig)

    def save_final_plot(self):
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
def evaluate_agent(agent, num_envs=10, num_episodes=20):
    """使用多个环境并行评估"""
    env_manager = ParallelEnvManager(num_envs=num_envs, easy_mode=False)
    success_count = 0
    total_rewards = []

    for ep in range(num_episodes):
        state = env_manager.reset_all()
        ep_r = 0
        steps = 0
        dones = np.zeros(num_envs, dtype=bool)

        while not np.all(dones):
            actions = agent.select_actions_batch(state, deterministic=True)
            next_state, rewards, terminateds, truncateds = env_manager.step_parallel(actions)
            ep_r += np.mean(rewards)
            state = next_state
            dones = terminateds | truncateds
            steps += 1

            if steps > 5000:
                break

        # 放宽成功条件
        success = False
        for i in range(num_envs):
            y_fm = env_manager.envs[i].y_fm
            if y_fm <= 0:  # 到达终点
                e_fm = env_manager.envs[i].x_fm - env_manager.envs[i].track_centerline(y_fm)
                theta_rel = env_manager.envs[i].theta - env_manager.envs[i].track_angle(y_fm)
                # 放宽阈值
                if abs(e_fm) < 0.1 and abs(theta_rel) < np.deg2rad(12):
                    success = True
                    break

        if success:
            success_count += 1
        total_rewards.append(ep_r)

    avg_r = np.mean(total_rewards)
    sr = success_count / num_episodes
    print(f"评估完成 | 平均奖励: {avg_r:.1f} | 成功率: {sr:.1%}")
    return {'success_rate': sr, 'avg_reward': avg_r}


# ========== 主训练函数 ==========
def train_sac_parallel(config):
    # 创建并行环境管理器
    env_manager = ParallelEnvManager(
        num_envs=config['num_envs'],
        easy_mode=config.get('easy_mode', False)
    )

    state_dim = 5
    action_dim = 2

    print("=" * 70)
    print("SAC 直升机牵引入库训练 - 并行环境版")
    print(f"并行环境数: {config['num_envs']}")
    print(f"状态维度: {state_dim} | 动作维度: {action_dim}")
    print(f"设备: {config['device']}")
    if config['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    agent = SACAgent(state_dim, action_dim, config)
    viz = TrainingVisualizer(config['save_dir'])

    start_time = time.time()
    best_success_rate = 0

    # 训练统计
    total_steps = 0
    episode = 0
    step_times = deque(maxlen=1000)

    # 收集初始经验
    print("\n收集初始经验...")
    state = env_manager.reset_all()
    for _ in range(config['initial_collect_steps']):
        actions = agent.select_actions_batch(state, deterministic=False)
        next_state, rewards, terminateds, truncateds = env_manager.step_parallel(actions)
        dones = terminateds | truncateds

        # 批量添加到buffer
        agent.buffer.push_batch(state, actions, rewards, next_state, dones)
        state = next_state
        total_steps += config['num_envs']

        # 重置完成的环境
        for i in range(config['num_envs']):
            if dones[i]:
                # 环境已自动重置
                pass

    print(f"初始经验收集完成，Buffer大小: {len(agent.buffer)}")

    # 训练循环
    episode_reward_sum = np.zeros(config['num_envs'])
    episode_length_sum = np.zeros(config['num_envs'])
    episode_count = np.zeros(config['num_envs'])

    while episode < config['max_episodes']:
        step_start = time.time()

        # 批量选择动作
        actions = agent.select_actions_batch(state, deterministic=(episode < 50))

        # 并行执行step
        next_state, rewards, terminateds, truncateds = env_manager.step_parallel(actions)
        dones = terminateds | truncateds

        # 批量添加到buffer
        agent.buffer.push_batch(state, actions, rewards, next_state, dones)

        # 更新累计奖励
        episode_reward_sum += rewards
        episode_length_sum += 1

        # 更新网络（多次）
        if len(agent.buffer) > config['batch_size']:
            for _ in range(config['updates_per_step']):
                agent.update()

        # 处理完成的episode
        for i in range(config['num_envs']):
            if dones[i]:
                episode += 1
                ep_reward = episode_reward_sum[i]
                ep_length = episode_length_sum[i]

                # 检查是否成功
                y_fm = env_manager.envs[i].y_fm
                e_fm = env_manager.envs[i].x_fm - env_manager.envs[i].track_centerline(y_fm)
                theta_rel = env_manager.envs[i].theta - env_manager.envs[i].track_angle(y_fm)
                success = y_fm <= 0 and abs(e_fm) < 0.1 and abs(theta_rel) < np.deg2rad(12)

                viz.update(episode, ep_reward, ep_length, success)

                # 重置该环境的累计
                episode_reward_sum[i] = 0
                episode_length_sum[i] = 0

                # 定期输出日志
                # 定期输出日志（每10个episode输出一次详细数据）
                if episode % config['log_interval'] == 0:
                    avg_reward = np.mean(viz.episode_rewards[-100:]) if len(viz.episode_rewards) >= 100 else np.mean(
                        viz.episode_rewards)
                    current_sr = viz.success_rates[-1] if viz.success_rates else 0
                    elapsed = (time.time() - start_time) / 60

                    # 计算步数/秒
                    steps_per_sec = config['num_envs'] / (time.time() - step_start + 0.001)

                    # 计算最近10个episode的平均数据
                    last_10_rewards = viz.episode_rewards[-10:] if len(
                        viz.episode_rewards) >= 10 else viz.episode_rewards
                    last_10_success = viz.success_rates[-10:] if len(viz.success_rates) >= 10 else viz.success_rates

                    print(f"\n{'=' * 80}")
                    print(f"Episode {episode:5d} 训练数据")
                    print(f"{'=' * 80}")
                    print(f"  本 episode:")
                    print(f"    奖励:        {ep_reward:8.1f}")
                    print(f"    步数:        {int(ep_length):8d}")
                    print(f"    成功:        {success}")
                    print(f"    最终偏差:    {e_fm:8.4f} m")
                    print(f"    最终偏角:    {np.rad2deg(theta_rel):6.1f}°")
                    print(f"    最终y位置:   {y_fm:6.2f} m")
                    print(f"")
                    print(f"  统计 (最近100轮):")
                    print(f"    平均奖励:    {avg_reward:8.1f}")
                    print(f"    成功率:      {current_sr:6.1%}")
                    print(f"")
                    print(f"  统计 (最近10轮):")
                    print(f"    平均奖励:    {np.mean(last_10_rewards):8.1f}")
                    print(f"    成功率:      {np.mean(last_10_success):6.1%}")
                    print(f"")
                    print(f"  性能:")
                    print(f"    环境步数/秒: {steps_per_sec:6.0f}")
                    print(f"    Buffer大小:  {len(agent.buffer):8d}")
                    print(f"    训练时间:    {elapsed:6.1f} min")
                    print(f"{'=' * 80}\n")

                # 保存最佳模型
                # 获取当前成功率
                current_sr = viz.success_rates[-1] if viz.success_rates else 0
                if current_sr > best_success_rate:
                    best_success_rate = current_sr
                    agent.save_model(f"{config['save_dir']}/best_model.pt")

                if episode >= config['max_episodes']:
                    break

        state = next_state
        total_steps += config['num_envs']

        step_times.append(time.time() - step_start)

        # 定期评估
        if episode > 0 and episode % config['eval_interval'] == 0:
            print("\n--- 评估中 ---")
            eval_result = evaluate_agent(agent, num_envs=10, num_episodes=20)
            print(f"评估结果: 成功率={eval_result['success_rate']:.1%}")
            print("--- 评估结束 ---\n")

    total_time = (time.time() - start_time) / 60
    print(f"\n训练完成！总耗时: {total_time:.1f} min")
    print(f"总步数: {total_steps}")
    print(f"平均步数/秒: {total_steps / (total_time * 60):.1f}")

    agent.save_model(f"{config['save_dir']}/final_model.pt")
    viz.save_final_plot()

    final_eval = evaluate_agent(agent, num_envs=10, num_episodes=50)
    return agent, final_eval


# ========== 主程序 ==========
if __name__ == "__main__":
    # 根据CPU核心数调整并行环境数
    cpu_count = mp.cpu_count()
    recommended_envs = min(cpu_count * 2, 32)  # 每个核心2个环境

    config = {
        'gamma': 0.99,
        'tau': 0.005,
        'alpha_lr': 1e-3,
        'hidden_dim': 256,
        'actor_lr': 3e-4,  # 降低学习率
        'critic_lr': 3e-4,

        'max_episodes': 3000,  # 增加训练轮数
        'batch_size': 1024,
        'buffer_capacity': 1000000,
        'initial_collect_steps': 10000,
        'updates_per_step': 2,

        'num_envs': 32,
        'easy_mode': False,

        # 新增：SAC温度参数
        'initial_alpha': 0.2,  # 初始探索率
        'target_entropy': -2.0,  # 目标熵（动作维度=2）

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': f'SAC_Parallel_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 20,
        'eval_interval': 200,
    }

    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存配置
    import json

    with open(f"{config['save_dir']}/config.json", 'w') as f:
        config_save = {k: str(v) if k == 'device' else v for k, v in config.items()}
        json.dump(config_save, f, indent=4)

    print(f"\n检测到CPU核心数: {cpu_count}")
    print(f"使用并行环境数: {config['num_envs']}")

    try:
        agent, result = train_sac_parallel(config)
        print("\n" + "=" * 70)
        print(f"训练全部完成！最终成功率: {result['success_rate']:.1%}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n训练被手动中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()