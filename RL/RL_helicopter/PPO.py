"""
基于PPO算法的直升机自动牵引入库训练代码（修复版）
使用运动学环境，符合论文第2.2节模型
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime
from helicopter_env import HelicopterInboundKinematicsEnv


# ========== 神经网络定义 ==========
class ActorNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # 动作均值（连续动作）
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )

        # 动作标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # 动作边界
        self.register_buffer('action_low', torch.tensor([-1.0, 0.05]))
        self.register_buffer('action_high', torch.tensor([1.0, 0.5]))

    def forward(self, state):
        shared_out = self.shared(state)
        mean = self.mean_layer(shared_out)

        # 将mean缩放到动作空间
        mean_scaled = self.action_low + (mean + 1) / 2 * (self.action_high - self.action_low)

        std = torch.exp(self.log_std)
        # 限制标准差最小值
        std = torch.clamp(std, min=0.01)
        return mean_scaled, std

    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        return self.net(state)


# ========== PPO算法实现 ==========
class PPOAgent:
    """PPO (Proximal Policy Optimization) 算法"""

    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device(config['device'])

        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.critic = CriticNetwork(state_dim, config['hidden_dim']).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])

        # 存储轨迹数据
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            value = self.critic(state)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0, 0]

    def store_transition(self, state, action, log_prob, reward, done, value):
        """存储轨迹数据"""
        self.states.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.actions.append(action.copy() if isinstance(action, np.ndarray) else action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value):
        """计算GAE (Generalized Advantage Estimation)"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_value):
        """更新网络参数"""
        if len(self.states) == 0:
            return

        # 计算优势函数和目标值
        advantages, returns = self.compute_gae(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化优势函数
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮更新
        for _ in range(self.config['update_epochs']):
            # 评估当前策略
            log_probs, entropy = self.actor.evaluate(states, actions)
            values = self.critic(states).squeeze()

            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'],
                                1 + self.config['clip_epsilon']) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 熵奖励（鼓励探索）
            actor_loss -= self.config['entropy_coef'] * entropy.mean()

            # Critic损失
            critic_loss = nn.MSELoss()(values, returns)

            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
            self.actor_optimizer.step()

            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
            self.critic_optimizer.step()

            self.training_losses.append({
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
            })

        # 清空缓冲区
        self.clear_buffer()

    def clear_buffer(self):
        """清空轨迹缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"模型已加载: {path}")


# ========== 训练环境包装器（修复版） ==========
class TrainingWrapper(gym.Wrapper):
    """训练环境包装器，添加噪声和奖励塑形"""

    def __init__(self, env, config):
        super().__init__(env)  # 使用gym.Wrapper自动处理observation_space和action_space
        self.config = config
        self.action_noise = config.get('action_noise', 0.05)
        self.last_action = np.zeros(2)

    def step(self, action):
        # 添加动作噪声（增加探索）
        if self.config.get('use_action_noise', True):
            noise = np.random.normal(0, self.action_noise, size=action.shape)
            action = np.clip(action + noise,
                             self.action_space.low,
                             self.action_space.high)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 奖励塑形（基于论文控制目标）
        e_fm, theta_rel, e_p, tail_angle, y_remaining = obs

        # 额外惩罚：不必要的横移（浪费能量）
        if abs(action[0]) > 0.5 and abs(e_fm) < 0.05 and abs(theta_rel) < 0.05:
            reward -= 0.5

        # 额外奖励：平滑控制（减少抖振）
        action_change = np.abs(action - self.last_action).sum()
        if action_change < 0.1:
            reward += 0.05
        self.last_action = action.copy()

        # 时间惩罚（鼓励快速完成）
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

        # 每10个episode更新图表
        if episode % 10 == 0:
            self._plot(episode)

    def _plot(self, episode):
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(self.episode_rewards)
        self.axes[0, 0].set_title('Episode Reward')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True)

        # 滑动平均
        if len(self.episode_rewards) > 50:
            if len(self.episode_rewards) >= 50:
                smoothed = np.convolve(self.episode_rewards, np.ones(50) / 50, mode='valid')
            self.axes[0, 0].plot(range(50, len(self.episode_rewards) + 1),
                                 smoothed, 'r', linewidth=2, label='Moving Avg')
            self.axes[0, 0].legend()

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

        # 显示统计信息
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
        else:
            avg_reward = np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
        stats_text = f"""
        Episode: {episode}
        Best Reward: {max(self.episode_rewards[-100:]):.1f}
        Avg Reward (100): {avg_reward:.1f}
        Success Rate: {self.success_rates[-1]:.2%}
        Avg Episode Length: {np.mean(self.episode_lengths[-50:]):.1f}
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
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            state = next_state

            if render and episode == num_episodes - 1:
                env.render()

        # 判断是否成功（到达机库且位姿正确）
        if hasattr(env, 'env'):
            y_fm = env.env.y_fm
        else:
            y_fm = env.y_fm

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
def train_ppo(config):
    """PPO训练主函数"""

    # 创建环境
    base_env = HelicopterInboundKinematicsEnv(render_mode=None)
    env = TrainingWrapper(base_env, config)

    state_dim = env.observation_space.shape[0]  # 5维状态
    action_dim = env.action_space.shape[0]  # 2维动作

    # 创建智能体
    agent = PPOAgent(state_dim, action_dim, config)

    # 可视化器
    visualizer = TrainingVisualizer(save_dir=config['save_dir'])

    # 训练记录
    best_reward = -np.inf
    training_start_time = time.time()

    print("=" * 60)
    print("开始训练 PPO 算法")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"设备: {agent.device}")
    print(f"保存目录: {config['save_dir']}")
    print("=" * 60)

    for episode in range(1, config['max_episodes'] + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        # 收集轨迹
        while not (terminated or truncated):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.store_transition(state, action, log_prob, reward, terminated or truncated, value)

            state = next_state
            episode_reward += reward
            episode_length += 1

            # 达到轨迹长度限制时更新
            if episode_length >= config['trajectory_length']:
                # 计算下一个状态的价值
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    next_value = agent.critic(next_state_tensor).cpu().numpy()[0, 0]
                agent.update(next_value)
                episode_length = 0

        # 处理最后一个不完整的轨迹
        if episode_length > 0:
            # 对于终止状态，下一个状态的价值为0
            if terminated:
                next_value = 0
            else:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    next_value = agent.critic(next_state_tensor).cpu().numpy()[0, 0]
            agent.update(next_value)

        # 记录统计
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)

        # 判断是否成功
        if hasattr(env, 'env'):
            y_fm = env.env.y_fm
            success = y_fm <= 0 and terminated
        else:
            success = False

        visualizer.update(episode, episode_reward, episode_length, success)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f"{config['save_dir']}/best_model.pt")

        # 定期打印训练信息
        if episode % config['log_interval'] == 0:
            rewards_list = list(agent.episode_rewards)[-100:]
            avg_reward = np.mean(rewards_list) if len(rewards_list) > 0 else 0
            lengths_list = list(agent.episode_lengths)[-100:]
            avg_length = np.mean(lengths_list) if len(lengths_list) > 0 else 0
            elapsed_time = time.time() - training_start_time

            print(f"Episode {episode}/{config['max_episodes']} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward (100): {avg_reward:.1f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success: {success} | "
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
        action, _, _ = agent.select_action(state, deterministic=True)
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

    # 训练配置（基于论文参数）
    config = {
        # 环境参数
        'action_noise': 0.03,  # 动作噪声
        'use_action_noise': True,  # 是否使用噪声

        # PPO超参数
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE参数
        'clip_epsilon': 0.2,  # PPO裁剪参数
        'entropy_coef': 0.01,  # 熵系数
        'max_grad_norm': 0.5,  # 梯度裁剪

        # 网络参数
        'hidden_dim': 256,  # 隐藏层维度
        'actor_lr': 3e-4,  # Actor学习率
        'critic_lr': 1e-3,  # Critic学习率

        # 训练参数
        'max_episodes': 100000,  # 最大训练episode数
        'trajectory_length': 2048,  # 轨迹收集长度
        'update_epochs': 10,  # 每轮更新次数

        # 其他
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 20,  # 打印间隔
        'eval_interval': 100,  # 评估间隔
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存配置
    import json

    with open(f"{config['save_dir']}/config.json", 'w') as f:
        # 转换numpy类型为Python原生类型
        config_copy = config.copy()
        json.dump(config_copy, f, indent=4)

    # 开始训练
    try:
        trained_agent, eval_results = train_ppo(config)

        print("\n" + "=" * 50)
        print("训练完成！")
        print(f"最终成功率: {eval_results['success_rate']:.2%}")
        print(f"模型保存在: {config['save_dir']}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback

        traceback.print_exc()