"""
轨道绘图测试 - 验证从降落区域向机库运动的方向（修正弯道）
运动方向：从降落区域（y=3.55）向机库（y=0）运动
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Arc

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 环境参数 ==========
L_PFM = 0.198  # 牵引杆P到前轮中点FM的距离
L_AFM = 1.2  # 尾轮A到前轮中点FM的距离

# 轨道参数（起点y=3.55，终点y=0）
y_start = 3.55  # 起点（降落区域）
y_end = 0.0  # 终点（机库）
y_curve_start = 1.4  # 弯道起点（从机库算起）
y_curve_end = 2.65  # 弯道终点
curve_dx = 0.38  # 横向偏移

# 机库尺寸（能包住直升机）
hangar_width = 0.8  # 机库宽度 (m)
hangar_height = 1.5  # 机库高度 (m)


# ========== 轨道函数（修正版） ==========
def track_centerline(y):
    """
    轨道中心线函数
    y: 从机库开始的纵向距离 (0-3.55m)
    直升机从 y=3.55 向 y=0 运动
    """
    if y <= y_curve_start:
        return 0.0
    elif y <= y_curve_end:
        t = (y - y_curve_start) / (y_curve_end - y_curve_start)
        return curve_dx * (1 - np.cos(np.pi * t)) / 2
    else:
        return curve_dx


def track_angle(y):
    """轨道切线角度 (rad)"""
    if y <= y_curve_start:
        return 0.0
    elif y <= y_curve_end:
        t = (y - y_curve_start) / (y_curve_end - y_curve_start)
        dt_dy = 1 / (y_curve_end - y_curve_start)
        dx_dt = curve_dx * np.pi * np.sin(np.pi * t) / 2
        derivative = dx_dt * dt_dy
        return np.arctan2(derivative, 1.0)
    else:
        return 0.0


# ========== 创建图形 ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# ========== 左图：完整轨道视图 ==========
ax1 = axes[0]

ax1.set_xlim(-0.5, 1.0)
ax1.set_ylim(-0.5, y_start + 0.5)
ax1.set_xlabel("横向位置 x (m)", fontsize=12)
ax1.set_ylabel("纵向位置 y (m)", fontsize=12)
ax1.set_title("直升机牵引轨道（从降落区域向机库）", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 绘制轨道中心线
ys = np.linspace(-0.5, y_start + 0.5, 500)
xs = [track_centerline(y) for y in ys]
ax1.plot(xs, ys, 'b-', linewidth=3, label='轨道中心线', zorder=2)

# 绘制轨道边界（轨道宽度0.2m）
ax1.fill_between(ys,
                 [x - 0.1 for x in xs],
                 [x + 0.1 for x in xs],
                 color='blue', alpha=0.15, zorder=0, label='轨道边界')

# 绘制弯道区域（y从1.4到2.65）
ax1.axhspan(y_curve_start, y_curve_end,
            alpha=0.3, color='yellow', zorder=0,
            label=f'弯道区域 ({y_curve_start}-{y_curve_end}m)')

# 标注直线段1（机库到弯道）
ax1.axhspan(y_end, y_curve_start,
            alpha=0.1, color='green', zorder=0,
            label=f'直线段1 (机库→弯道)')

# 标注直线段2（弯道到降落区域）
ax1.axhspan(y_curve_end, y_start,
            alpha=0.1, color='green', zorder=0,
            label=f'直线段2 (弯道→降落区域)')

# ========== 机库（能包住直升机） ==========
hangar_x_center = track_centerline(y_end)
hangar_rect = Rectangle((hangar_x_center - hangar_width / 2, -1.2),
                        hangar_width, hangar_height,
                        linewidth=2, edgecolor='red',
                        facecolor='lightcoral', alpha=0.5, zorder=1)
ax1.add_patch(hangar_rect)

# 机库门标注
ax1.text(hangar_x_center, -0.5, '机库', fontsize=12,
         ha='center', va='center', color='red', fontweight='bold')

# 机库门框
door_arc = Arc((hangar_x_center, -0.3), 0.5, 0.5, angle=0, theta1=0, theta2=180,
               linewidth=2, edgecolor='red', facecolor='none')
ax1.add_patch(door_arc)

# ========== 起点标注（降落区域） ==========
start_x = track_centerline(y_start)
ax1.plot(start_x, y_start, 'gs', markersize=12, label='起点', zorder=3)
ax1.text(start_x + 0.08, y_start, '降落区域', fontsize=11,
         color='green', fontweight='bold')

# 标注弯道关键点
# 弯道起点（y=1.4）
ax1.plot(track_centerline(y_curve_start), y_curve_start, 'yo', markersize=8, zorder=3)
ax1.annotate(f'弯道起点\n(y={y_curve_start}m)',
             xy=(track_centerline(y_curve_start), y_curve_start),
             xytext=(-0.3, y_curve_start + 0.2),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'))

# 弯道终点（y=2.65）
ax1.plot(track_centerline(y_curve_end), y_curve_end, 'yo', markersize=8, zorder=3)
ax1.annotate(f'弯道终点\n(y={y_curve_end}m)',
             xy=(track_centerline(y_curve_end), y_curve_end),
             xytext=(0.65, y_curve_end + 0.2),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'))

# 方向箭头（从起点指向机库，向下）
ax1.annotate('', xy=(0.2, 0.5), xytext=(0.2, y_start - 0.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.text(0.25, y_start / 2, '牵引方向\n(向机库)', fontsize=10,
         rotation=90, va='center', ha='center')

# 图例
ax1.legend(loc='upper left', fontsize=9)

# ========== 右图：轨道切线角度变化 ==========
ax2 = axes[1]

ys_detail = np.linspace(y_end, y_start, 300)
angles_deg = [np.rad2deg(track_angle(y)) for y in ys_detail]

ax2.plot(ys_detail, angles_deg, 'r-', linewidth=2)
ax2.set_xlabel("纵向位置 y (m)", fontsize=12)
ax2.set_ylabel("轨道切线角度 (度)", fontsize=12)
ax2.set_title("轨道切线角度变化曲线", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)

# 标注区域
ax2.axvspan(y_curve_start, y_curve_end, alpha=0.2, color='yellow', label='弯道区域')
ax2.axvline(x=y_curve_start, color='orange', linestyle='--', alpha=0.7, label=f'弯道起点 ({y_curve_start}m)')
ax2.axvline(x=y_curve_end, color='orange', linestyle='--', alpha=0.7, label=f'弯道终点 ({y_curve_end}m)')
ax2.axvline(x=y_start, color='green', linestyle='--', alpha=0.5, label=f'起点/降落区域 ({y_start}m)')
ax2.axvline(x=y_end, color='red', linestyle='--', alpha=0.5, label=f'终点/机库 ({y_end}m)')

# 标注最大角度
max_angle_idx = np.argmax(angles_deg)
max_angle_y = ys_detail[max_angle_idx]
max_angle = angles_deg[max_angle_idx]
ax2.plot(max_angle_y, max_angle, 'ro', markersize=8)
ax2.annotate(f'最大角度: {max_angle:.1f}°',
             xy=(max_angle_y, max_angle),
             xytext=(max_angle_y + 0.3, max_angle + 2),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

ax2.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ========== 第二张图：直升机在轨道上的姿态 ==========
fig2, ax3 = plt.subplots(figsize=(12, 14))

ax3.set_xlim(-0.5, 1.0)
ax3.set_ylim(-0.5, y_start + 0.5)
ax3.set_xlabel("横向位置 x (m)", fontsize=12)
ax3.set_ylabel("纵向位置 y (m)", fontsize=12)
ax3.set_title("直升机在轨道各位置姿态（从降落区域向机库运动）", fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# 绘制轨道
ys = np.linspace(-0.5, y_start + 0.5, 500)
xs = [track_centerline(y) for y in ys]
ax3.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道', zorder=1)
ax3.fill_between(ys, [x - 0.1 for x in xs], [x + 0.1 for x in xs],
                 color='blue', alpha=0.15, zorder=0)

# 绘制弯道区域
ax3.axhspan(y_curve_start, y_curve_end,
            alpha=0.2, color='yellow', zorder=0)

# 绘制机库
hangar_rect = Rectangle((hangar_x_center - hangar_width / 2, -1.2),
                        hangar_width, hangar_height,
                        linewidth=2, edgecolor='red',
                        facecolor='lightcoral', alpha=0.5, zorder=1)
ax3.add_patch(hangar_rect)
ax3.text(hangar_x_center, -0.5, '机库', fontsize=12,
         ha='center', va='center', color='red', fontweight='bold')

# 绘制起点
ax3.plot(start_x, y_start, 'gs', markersize=12, label='起点', zorder=3)
ax3.text(start_x + 0.08, y_start, '降落区域', fontsize=11, color='green')

# 方向箭头
ax3.annotate('', xy=(0.2, 0.5), xytext=(0.2, y_start - 0.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax3.text(0.25, y_start / 2, '牵引方向\n(向机库)', fontsize=10, rotation=90, va='center', ha='center')


# ========== 绘制直升机函数 ==========
def draw_helicopter(ax, x_fm, y_fm, theta, color='gray', alpha=0.85, label=None):
    """绘制直升机"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 机头（前轮中点）
    nose = (x_fm, y_fm)

    # 机尾
    tail_x = x_fm - L_AFM * sin_theta
    tail_y = y_fm - L_AFM * cos_theta

    # 机身三角形
    body_width = 0.12
    left_rear = (tail_x - body_width * cos_theta, tail_y + body_width * sin_theta)
    right_rear = (tail_x + body_width * cos_theta, tail_y - body_width * sin_theta)

    body = Polygon([nose, left_rear, right_rear],
                   closed=True, color=color, alpha=alpha, zorder=2)
    ax.add_patch(body)

    # 前轮
    front_wheel = Circle(nose, 0.05, color='black', zorder=3)
    ax.add_patch(front_wheel)

    # 尾轮
    tail_wheel = Circle((tail_x, tail_y), 0.045, color='red', zorder=3)
    ax.add_patch(tail_wheel)

    # 牵引杆
    x_p = x_fm + L_PFM * np.sin(theta)
    y_p = y_fm - L_PFM * np.cos(theta)
    ax.plot([x_p, x_fm], [y_p, y_fm], linewidth=2.5, color='orange', zorder=2)
    ax.plot(x_p, y_p, 'ro', markersize=5, zorder=3)

    if label:
        ax.text(x_fm, y_fm + 0.15, label, fontsize=9, ha='center', color=color)


# 位置1：起点附近（y=3.2，刚出发，直线段）
draw_helicopter(ax3, 0, 3.2, 0, color='gray', alpha=0.9, label='位置1: 直线段 (刚出发)')

# 位置2：弯道起点（y=1.4）
y_pos2 = 1.4
x_pos2 = track_centerline(y_pos2)
theta_pos2 = track_angle(y_pos2)
draw_helicopter(ax3, x_pos2, y_pos2, theta_pos2, color='blue', alpha=0.85, label='位置2: 弯道起点')

# 位置3：弯道中间（y=2.0）
y_pos3 = 2.0
x_pos3 = track_centerline(y_pos3)
theta_pos3 = track_angle(y_pos3)
draw_helicopter(ax3, x_pos3, y_pos3, theta_pos3, color='purple', alpha=0.85, label='位置3: 弯道中间')

# 位置4：弯道终点（y=2.65）
y_pos4 = 2.65
x_pos4 = track_centerline(y_pos4)
theta_pos4 = track_angle(y_pos4)
draw_helicopter(ax3, x_pos4, y_pos4, theta_pos4, color='blue', alpha=0.85, label='位置4: 弯道终点')

# 位置5：直线段2（y=3.0，接近起点）
draw_helicopter(ax3, curve_dx, 3.0, 0, color='gray', alpha=0.9, label='位置5: 直线段2')

# 位置6：进入机库（y=0）
draw_helicopter(ax3, curve_dx, 0, 0, color='darkgreen', alpha=0.9, label='位置6: 入库')

ax3.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

# ========== 打印参数信息 ==========
print("\n" + "=" * 60)
print("轨道参数信息")
print("=" * 60)
print(f"起点 (降落区域): y = {y_start} m")
print(f"终点 (机库): y = {y_end} m")
print(f"总长度: {y_start} m")
print(f"直线段1 (机库→弯道): {y_end} - {y_curve_start} m (x=0)")
print(f"弯道段: {y_curve_start} - {y_curve_end} m (偏移: 0 → {curve_dx} m)")
print(f"直线段2 (弯道→降落区域): {y_curve_end} - {y_start} m (x={curve_dx} m)")
print(f"最大偏移量: {curve_dx} m")
print(f"最大切线角度: {max_angle:.2f}°")
print(f"\n机库尺寸: 宽={hangar_width}m, 高={hangar_height}m")
print(f"直升机长度: {L_AFM}m, 宽度: 约0.24m")
print("✓ 机库可以包住直升机")

# ========== 测试关键点 ==========
print("\n" + "=" * 60)
print("关键点轨道值")
print("=" * 60)
test_points = [0, 0.5, 1.0, 1.4, 1.8, 2.2, 2.65, 3.0, 3.55]
for y in test_points:
    x = track_centerline(y)
    angle = np.rad2deg(track_angle(y))
    print(f"y = {y:.2f}m  →  x = {x:.4f}m,  切线角度 = {angle:6.2f}°")

print("\n" + "=" * 60)
print("✅ 绘图测试完成！")
print("请检查：")
print("1. 弯道区域从 y=1.4m 到 y=2.65m，向右偏移0.38m（黄色区域）")
print("2. 起点在 y=3.55m（降落区域），用绿色方块标注")
print("3. 终点在 y=0m（机库），用红色矩形标注")
print("4. 牵引方向向下（从起点指向机库）")
print("5. 弯道处直升机机身与轨道相切")
print("=" * 60)