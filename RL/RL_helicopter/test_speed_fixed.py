# test_speed_fixed.py
import numpy as np
from helicopter_env import HelicopterInboundKinematicsEnv

# 临时修改max_steps避免超时
env = HelicopterInboundKinematicsEnv(render_mode=None)
env.max_steps = 500

print("=" * 50)
print("速度测试 - 使用修正后的运动学公式")
print(f"理论最大速度: {env.VY_MAX}m/s")
print(f"理论到达步数: {env.y_start / env.VY_MAX / 0.05:.0f}")
print("=" * 50)

for test in range(3):
    obs, _ = env.reset()
    y_start = obs[4]
    print(f"\n测试 {test + 1}: 起始 y={y_start:.2f}")

    # 只向前，不横移
    for step in range(300):
        action = np.array([0.0, env.VY_MAX])  # 最大速度向前
        obs, reward, terminated, truncated, _ = env.step(action)

        if step % 50 == 0:
            print(f"  Step {step}: y={obs[4]:.2f}")

        if terminated:
            print(f"✓ 到达机库！步数: {step + 1}")
            break

    y_end = obs[4]
    distance = y_start - y_end
    actual_speed = distance / (300 * 0.05)
    print(f"结束 y={y_end:.2f}, 移动距离={distance:.2f}m, 实际速度={actual_speed:.4f}m/s")

env.close()

