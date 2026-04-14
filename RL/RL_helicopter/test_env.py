"""
测试基本控制逻辑
"""

import numpy as np
from helicopter_env import HelicopterInboundKinematicsEnv


def test_pid_control():
    """测试PID控制器的性能"""

    env = HelicopterInboundKinematicsEnv(render_mode="human")

    class SimplePID:
        def __init__(self, kp=1.0):
            self.kp = kp

        def compute(self, error):
            return np.clip(self.kp * error, -1, 1)

    pid = SimplePID(kp=1.2)

    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1} 开始...")

        while True:
            e_fm, theta_rel, e_p, tail_angle, y_remaining = state

            # PID控制
            vx_sign = pid.compute(-e_fm)

            # 简单的速度控制
            if abs(e_fm) < 0.15 and abs(theta_rel) < 0.1:
                vy = 0.3
            elif abs(e_fm) > 0.4 or abs(theta_rel) > 0.3:
                vy = 0.08
            else:
                vy = 0.2

            action = np.array([vx_sign, vy])
            state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

            env.render()

            if terminated or truncated or steps > 800:
                break

        success = terminated and abs(env.x_fm - env.track_centerline(env.y_fm)) < 0.3
        print(f"Episode {episode + 1}: {'成功' if success else '失败'}, "
              f"步数={steps}, 奖励={total_reward:.1f}")

    env.close()


def test_manual_control():
    """手动测试控制"""

    env = HelicopterInboundKinematicsEnv(render_mode="human")

    for episode in range(5):
        state, _ = env.reset()

        print(f"\nEpisode {episode + 1} - 使用固定控制策略")

        for step in range(500):
            e_fm, theta_rel, e_p, tail_angle, y_remaining = state

            # 固定策略
            if e_fm > 0.1:
                vx_sign = -1  # 向左
            elif e_fm < -0.1:
                vx_sign = 1  # 向右
            else:
                vx_sign = 0  # 不动

            vy = 0.2  # 恒定速度

            action = np.array([vx_sign, vy])
            state, reward, terminated, truncated, _ = env.step(action)

            env.render()

            if terminated or truncated:
                break

        print(f"最终位置: y={env.y_fm:.1f}, 偏差={state[0]:.3f}")

    env.close()


if __name__ == "__main__":
    print("测试PID控制...")
    test_pid_control()

    print("\n测试手动控制...")
    test_manual_control()
