"""
轨道可视化
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def track_centerline(y):
    """轨道中心线函数"""
    y_start = 0.0  # 机库位置
    y_end = 30.0  # 起点位置
    y_curve_start = 8.0
    y_curve_end = 22.0
    curve_dx = 0.6

    if y <= y_curve_start:
        return 0.0
    elif y <= y_curve_end:
        t = (y - y_curve_start) / (y_curve_end - y_curve_start)
        return curve_dx * (1 - np.cos(np.pi * t)) / 2
    else:
        return curve_dx


def track_angle(y):
    """轨道切线角度"""
    y_curve_start = 8.0
    y_curve_end = 22.0
    curve_dx = 0.6

    if y <= y_curve_start:
        return 0.0
    elif y <= y_curve_end:
        t = (y - y_curve_start) / (y_curve_end - y_curve_start)
        dx_dt = curve_dx * np.pi * np.sin(np.pi * t) / 2
        dt_dy = 1 / (y_curve_end - y_curve_start)
        derivative = dx_dt * dt_dy
        return np.arctan2(derivative, 1.0)
    else:
        return 0.0


# 生成轨道点
ys = np.linspace(0, 30, 1000)
xs = [track_centerline(y) for y in ys]
angles = [np.rad2deg(track_angle(y)) for y in ys]

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 上子图：轨道形状
axes[0].plot(xs, ys, 'b-', linewidth=3)
axes[0].fill_between(ys,
                     [x - 0.1 for x in xs],
                     [x + 0.1 for x in xs],
                     color='blue', alpha=0.2)
axes[0].set_xlabel('横向位置 x (m)')
axes[0].set_ylabel('纵向位置 y (m)')
axes[0].set_title('牵引轨道形状')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=8, color='r', linestyle='--', alpha=0.5, label='弯道起点 (y=8m)')
axes[0].axhline(y=22, color='r', linestyle='--', alpha=0.5, label='弯道终点 (y=22m)')
axes[0].legend()
axes[0].set_aspect('equal')

# 下子图：轨道切线角度
axes[1].plot(ys, angles, 'g-', linewidth=2)
axes[1].set_xlabel('纵向位置 y (m)')
axes[1].set_ylabel('切线角度 (度)')
axes[1].set_title('轨道切线角度变化')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].axvline(x=8, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(x=22, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 打印轨道信息
print("=" * 50)
print("轨道参数信息")
print("=" * 50)
print(f"总长度: 30 m")
print(f"直线段1: 0-8 m (x=0)")
print(f"弯道段: 8-22 m (偏移: 0 → 0.6 m)")
print(f"直线段2: 22-30 m (x=0.6)")
print(f"最大曲率半径: 约 {14 / (0.6 * np.pi):.1f} m")
print(f"最大切线角度: {np.rad2deg(track_angle(15)):.1f}°")

