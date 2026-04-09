# test_direction.py
import numpy as np
from helicopter_env import HelicopterInboundKinematicsEnv

env = HelicopterInboundKinematicsEnv()

print("=" * 50)
print("方向测试：直升机应该从 y≈3.55 向 y=0 移动")
print("=" * 50)

for episode in range(3):
    obs, _ = env.reset()
    print(f"\nEpisode {episode + 1}, 起始 y={obs[4]:.2f}")

    for step in range(300):
        e_fm = obs[0]
        # 简单PID控制修正偏差
        vx_sign = -np.clip(e_fm * 5, -1, 1)
        vy = 0.1  # 正向速度应该向机库移动

        action = np.array([vx_sign, vy])
        obs, reward, terminated, truncated, _ = env.step(action)

        if step % 50 == 0:
            print(f"  Step {step}: y={obs[4]:.2f}, e={obs[0]:.3f}")

        env.render()

        if terminated or truncated:
            print(f"结束: y={obs[4]:.2f}, 成功={terminated}")
            break

    if not (terminated or truncated):
        print(f"超时: y={obs[4]:.2f}")

env.close()