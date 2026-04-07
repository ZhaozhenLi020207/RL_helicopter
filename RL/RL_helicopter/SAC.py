"""
基于SAC (Soft Actor-Critic)算法的直升机自动牵引入库训练
SAC特点：最大熵框架、探索能力强、训练稳定
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
from helicopter_env import HelicopterInboundKinematicsEnv


# ========== 网络定义 ==========
class SoftQNetwork(nn.Module):
    """Soft Q网络（Critic）"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """高斯策略网络（Actor）- SAC使用重参数化技巧"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()

        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            # 重参数化采样
            z = normal.rsample()  # rsample支持梯度回传
            action = torch.tanh(z)  # 限制动作范围[-1, 1]

            # 计算log概率（考虑tanh变换）
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 缩放到实际动作空间
        action = action * self.action_scale + self.action_bias

        return action, log_prob

    def evaluate(self, state, action):
        """评估给定动作的log概率"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 将动作反归一化到[-1, 1]范围
        action_normalized = (action - self.action_bias) / self.action_scale
        action_normalized = torch.clamp(action_normalized, -0.999, 0.999)

        # 计算反tanh
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

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


# ========== SAC Agent ==========
class SACAgent:
    """Soft Actor-Critic Agent"""

    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device(config['device'])

        # 动作空间边界
        self.action_low = np.array([-1.0, 0.05])
        self.action_high = np.array([1.0, 0.5])
        self.action_scale = torch.FloatTensor((self.action_high - self.action_low) / 2).to(self.device)
        self.action_bias = torch.FloatTensor((self.action_high + self.action_low) / 2).to(self.device)

        # 创建网络
        self.actor = GaussianPolicy(
            state_dim, action_dim,
            config['hidden_dim'],
            self.action_scale, self.action_bias
        ).to(self.device)

        # 两个Q网络（减少过估计）
        self.q1 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)

        # 复制参数到目标网络
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config['critic_lr'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config['critic_lr'])

        # 自动调节温度系数α
        self.target_entropy = -action_dim  # H* = -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['alpha_lr'])

        # Replay buffer
        self.buffer = ReplayBuffer(
            config['buffer_capacity'],
            state_dim,
            action_dim,
            self.device
        )

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic)

        return action.cpu().numpy()[0]

    def update(self):
        """更新网络参数"""
        if len(self.buffer) < self.config['batch_size']:
            return

        # 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config['batch_size']
        )

        with torch.no_grad():
            # 采样下一个动作（使用当前策略）
            next_actions, next_log_probs = self.actor.sample(next_states)

            # 计算目标Q值
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config['gamma'] * (1 - dones) * target_q

        # 更新Q网络
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

        # 更新Actor网络
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度系数α
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # 软更新目标网络
        tau = self.config['tau']
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 记录损失
        self.training_losses.append({
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        })

    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.alpha = self.log_alpha.exp()
        print(f"模型已加载: {path}")


# ========== 训练环境包装器 ==========
class TrainingWrapper(gym.Wrapper):
    """训练环境包装器"""

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.last_action = np.zeros(2)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 奖励塑形
        e_fm, theta_rel, e_p, tail_angle, y_remaining = obs

        # 平滑控制奖励
        action_change = np.abs(action - self.last_action).sum()
        if action_change < 0.1:
            reward += 0.05
        self.last_action = action.copy()

        # 时间惩罚
        reward -= 0.005

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(2)
        return obs, info


# ========== 训练可视化 ==========
class TrainingVisualizer:
    """训练过程可视化"""

    def __init__(self, save_dir='training_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.rolling_success = deque(maxlen=50)

    def update(self, episode, reward, length, success):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.rolling_success.append(1 if success else 0)
        self.success_rates.append(np.mean(self.rolling_success))

        if episode % 10 == 0:
            self._plot(episode)

    def _plot(self, episode):
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(self.episode_rewards)
        self.axes[0, 0].set_title('Episode Reward')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True)

        if len(self.episode_rewards) > 50:
            smoothed = np.convolve(self.episode_rewards, np.ones(50) / 50, mode='valid')
            self.axes[0, 0].plot(range(50, len(self.episode_rewards) + 1), smoothed, 'r', linewidth=2)

        self.axes[0, 1].clear()
        self.axes[0, 1].plot(self.episode_lengths)
        self.axes[0, 1].set_title('Episode Length')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].grid(True)

        self.axes[0, 2].clear()
        self.axes[0, 2].plot(self.success_rates)
        self.axes[0, 2].set_title('Success Rate (50-episode rolling)')
        self.axes[0, 2].set_xlabel('Episode')
        self.axes[0, 2].set_ylabel('Success Rate')
        self.axes[0, 2].set_ylim([0, 1])
        self.axes[0, 2].grid(True)

        self.axes[1, 0].clear()
        self.axes[1, 0].set_axis_off()
        self.axes[1, 1].clear()
        self.axes[1, 1].set_axis_off()
        self.axes[1, 2].clear()
        self.axes[1, 2].set_axis_off()

        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
            self.episode_rewards)
        stats_text = f"""
        Episode: {episode}
        Best Reward: {max(self.episode_rewards[-100:]):.1f}
        Avg Reward (100): {avg_reward:.1f}
        Success Rate: {self.success_rates[-1]:.2%}
        Avg Length: {np.mean(self.episode_lengths[-50:]):.1f}
        """
        self.axes[1, 0].text(0.1, 0.5, stats_text, transform=self.axes[1, 0].transAxes,
                             fontsize=12, verticalalignment='center',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png', dpi=150)

    def save_final_plot(self):
        self._plot(len(self.episode_rewards))
        plt.savefig(f'{self.save_dir}/final_training_plot.png', dpi=150)
        plt.close(self.fig)


# ========== 评估函数 ==========
def evaluate_agent(agent, env, num_episodes=10, render=False):
    """评估训练好的智能体"""
    success_count = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            state = next_state

            if render and episode == num_episodes - 1:
                env.render()

        # 判断是否成功
        y_fm = env.env.y_fm if hasattr(env, 'env') else env.y_fm
        success = y_fm <= 0 and terminated
        if success:
            success_count += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        print(f"评估 Episode {episode + 1}: Reward={episode_reward:.1f}, "
              f"Length={step_count}, Success={success}")

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes

    print(f"\n评估结果 (共{num_episodes}个episode):")
    print(f"平均奖励: {avg_reward:.1f}")
    print(f"平均长度: {avg_length:.1f}")
    print(f"成功率: {success_rate:.2%}")

    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'rewards': total_rewards,
        'lengths': episode_lengths
    }


# ========== 主训练函数 ==========
def train_sac(config):
    """SAC训练主函数"""

    # 创建环境
    base_env = HelicopterInboundKinematicsEnv(render_mode=None)
    env = TrainingWrapper(base_env, config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建智能体
    agent = SACAgent(state_dim, action_dim, config)

    # 可视化器
    visualizer = TrainingVisualizer(save_dir=config['save_dir'])

    # 训练记录
    best_reward = -np.inf
    training_start_time = time.time()

    print("=" * 60)
    print("开始训练 SAC (Soft Actor-Critic) 算法")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"设备: {agent.device}")
    print(f"保存目录: {config['save_dir']}")
    print("=" * 60)

    total_steps = 0

    for episode in range(1, config['max_episodes'] + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        # 收集轨迹
        while not (terminated or truncated):
            # 选择动作
            action = agent.select_action(state, deterministic=False)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 存储经验
            agent.buffer.push(state, action, reward, next_state, terminated or truncated)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # 更新网络
            if total_steps > config['learning_starts']:
                for _ in range(config['updates_per_step']):
                    agent.update()

        # 记录episode统计
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)

        # 判断是否成功
        y_fm = env.env.y_fm if hasattr(env, 'env') else env.y_fm
        success = y_fm <= 0 and terminated

        visualizer.update(episode, episode_reward, episode_length, success)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f"{config['save_dir']}/best_model.pt")

        # 定期打印
        if episode % config['log_interval'] == 0:
            recent_rewards = agent.episode_rewards[-100:] if len(
                agent.episode_rewards) >= 100 else agent.episode_rewards
            recent_lengths = agent.episode_lengths[-100:] if len(
                agent.episode_lengths) >= 100 else agent.episode_lengths
            avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
            avg_length = np.mean(recent_lengths) if len(recent_lengths) > 0 else 0
            elapsed_time = time.time() - training_start_time

            print(f"Episode {episode}/{config['max_episodes']} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward (100): {avg_reward:.1f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success: {success} | "
                  f"Alpha: {agent.alpha.item():.3f} | "
                  f"Buffer: {len(agent.buffer)} | "
                  f"Time: {elapsed_time / 60:.1f} min")

        # 定期评估
        if episode % config['eval_interval'] == 0:
            print("\n" + "-" * 40)
            print("开始评估...")
            eval_results = evaluate_agent(agent, env, num_episodes=10, render=False)
            print(f"评估结果: 成功率={eval_results['success_rate']:.2%}")
            print("-" * 40 + "\n")

    # 训练结束
    training_time = time.time() - training_start_time
    print(f"\n训练完成！总耗时: {training_time / 60:.1f} 分钟")

    # 保存最终模型
    agent.save_model(f"{config['save_dir']}/final_model.pt")
    visualizer.save_final_plot()

    # 最终评估
    print("\n最终评估 (50 episodes)...")
    final_eval = evaluate_agent(agent, env, num_episodes=50, render=False)

    # 生成演示动画
    print("\n生成演示动画...")
    demo_env = HelicopterInboundKinematicsEnv(render_mode=None)
    state, _ = demo_env.reset()
    demo_trajectory = []

    for _ in range(800):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = demo_env.step(action)
        demo_trajectory.append({
            'x_fm': demo_env.x_fm, 'y_fm': demo_env.y_fm,
            'theta': demo_env.theta, 'x_p': demo_env.x_p, 'y_p': demo_env.y_p,
            'tail_angle': demo_env.tail_angle
        })
        state = next_state
        if terminated or truncated:
            break

    demo_env.trajectory = demo_trajectory
    demo_env.make_animation(f"{config['save_dir']}/trained_agent_demo.gif", save_frames=200)
    demo_env.close()

    return agent, final_eval


# ========== 主程序 ==========
if __name__ == "__main__":

    # SAC训练配置
    config = {
        # SAC超参数
        'gamma': 0.99,  # 折扣因子
        'tau': 0.005,  # 目标网络软更新参数
        'alpha_lr': 3e-4,  # 温度系数学习率

        # 网络参数
        'hidden_dim': 256,  # 隐藏层维度
        'actor_lr': 3e-4,  # Actor学习率
        'critic_lr': 3e-4,  # Critic学习率

        # 训练参数
        'max_episodes': 2500,  # 最大训练episode数
        'batch_size': 256,  # 批量大小
        'buffer_capacity': 700000,  # 经验池容量
        'learning_starts': 10000,  # 开始学习前的步数
        'updates_per_step': 1,  # 每步更新次数

        # 其他
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': f'SAC_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 20,  # 打印间隔
        'eval_interval': 100,  # 评估间隔
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存配置
    import json

    with open(f"{config['save_dir']}/config.json", 'w') as f:
        config_copy = config.copy()
        # 转换非JSON可序列化的值
        config_copy['device'] = str(config_copy['device'])
        json.dump(config_copy, f, indent=4)

    # 开始训练
    try:
        trained_agent, eval_results = train_sac(config)

        print("\n" + "=" * 50)
        print("SAC训练完成！")
        print(f"最终成功率: {eval_results['success_rate']:.2%}")
        print(f"模型保存在: {config['save_dir']}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback

        traceback.print_exc()