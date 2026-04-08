"""
使用PID控制器收集专家数据
"""

import numpy as np
from helicopter_env import HelicopterInboundKinematicsEnv
import pickle
import os


class PIDController:
    """简单的PID控制器"""

    def __init__(self, kp=1.5, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error

        # 限制输出范围
        return np.clip(output, -1, 1)


def collect_expert_data(num_episodes=100, save_path="expert_data.pkl"):
    """收集专家数据"""

    env = HelicopterInboundKinematicsEnv(render_mode=None)

    # 创建PID控制器
    pid_e = PIDController(kp=1.2, ki=0.05, kd=0.3)  # 用于偏差
    pid_theta = PIDController(kp=0.8, ki=0.02, kd=0.2)  # 用于偏角

    expert_data = []
    success_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_data = []
        episode_reward = 0
        steps = 0

        pid_e.integral = 0
        pid_e.last_error = 0
        pid_theta.integral = 0
        pid_theta.last_error = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            e_fm, theta_rel, e_p, tail_angle, y_remaining = state

            # PID控制
            vx_cmd_e = pid_e.compute(-e_fm)  # 负号：偏差为正时向左移动
            vx_cmd_theta = pid_theta.compute(-theta_rel)

            # 综合控制
            vx_sign = np.clip((vx_cmd_e + vx_cmd_theta) / 2, -1, 1)

            # 速度策略
            if abs(e_fm) < 0.1 and abs(theta_rel) < 0.05:
                vy = 0.3  # 状态好时加速
            elif abs(e_fm) > 0.3 or abs(theta_rel) > 0.2:
                vy = 0.1  # 状态差时减速
            else:
                vy = 0.2  # 正常速度

            action = np.array([vx_sign, vy])

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': terminated or truncated
            })

            episode_reward += reward
            state = next_state
            steps += 1

            if steps > 500:
                break

        if terminated and abs(env.x_fm - env.track_centerline(env.y_fm)) < 0.2:
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS, steps={steps}, reward={episode_reward:.1f}")
            expert_data.extend(episode_data)
        else:
            print(f"Episode {episode + 1}: FAILED, steps={steps}, reward={episode_reward:.1f}")

        env.reset()  # 重置环境

    print(f"\n收集完成！成功率: {success_count / num_episodes:.2%}")
    print(f"总数据量: {len(expert_data)}")

    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump(expert_data, f)

    env.close()
    return expert_data


if __name__ == "__main__":
    collect_expert_data(num_episodes=200)
