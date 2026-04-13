"""
直升机入库轨道检测与可视化
检测轨道参数、曲率、切线角度等
"""

import numpy as np
import matplotlib.pyplot as plt
from helicopter_env import HelicopterInboundKinematicsEnv

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def analyze_track():
    """分析轨道特性"""
    env = HelicopterInboundKinematicsEnv(fast_mode=False)

    # 生成y坐标点（从终点到起点）
    y_values = np.linspace(-0.2, env.y_start + 0.2, 500)

    # 计算轨道参数
    centerline = []
    derivative = []
    angle = []
    curvature = []

    for y in y_values:
        x = env.track_centerline(y)
        dx_dy = env.track_derivative(y)
        theta = env.track_angle(y)

        centerline.append(x)
        derivative.append(dx_dy)
        angle.append(np.rad2deg(theta))

        # 计算曲率（近似）
        if len(centerline) > 1:
            dx = centerline[-1] - centerline[-2]
            dy = y_values[len(centerline)-1] - y_values[len(centerline)-2]
            if dy != 0:
                curv = abs(dx / dy)  # 简化曲率
            else:
                curv = 0
            curvature.append(curv)
        else:
            curvature.append(0)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('直升机入库轨道分析', fontsize=16, fontweight='bold')

    # 1. 轨道中心线
    ax1 = axes[0, 0]
    ax1.plot(y_values, centerline, 'b-', linewidth=2, label='轨道中心线')
    ax1.fill_between(y_values,
                      [x - 0.1 for x in centerline],
                      [x + 0.1 for x in centerline],
                      color='blue', alpha=0.2, label='轨道边界 (±0.1m)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='机库位置')
    ax1.axvline(x=env.y_curve_start, color='orange', linestyle='--', alpha=0.5, label=f'弯道起点 ({env.y_curve_start}m)')
    ax1.axvline(x=env.y_curve_end, color='orange', linestyle='--', alpha=0.5, label=f'弯道终点 ({env.y_curve_end}m)')
    ax1.set_xlabel('纵向位置 y (m)')
    ax1.set_ylabel('横向位置 x (m)')
    ax1.set_title('轨道中心线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 轨道导数 dx/dy
    ax2 = axes[0, 1]
    ax2.plot(y_values, derivative, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=env.y_curve_start, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(x=env.y_curve_end, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('纵向位置 y (m)')
    ax2.set_ylabel('dx/dy')
    ax2.set_title('轨道导数 (斜率)')
    ax2.grid(True, alpha=0.3)

    # 3. 轨道切线角度
    ax3 = axes[0, 2]
    ax3.plot(y_values, angle, 'r-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=env.y_curve_start, color='orange', linestyle='--', alpha=0.5)
    ax3.axvline(x=env.y_curve_end, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('纵向位置 y (m)')
    ax3.set_ylabel('角度 (度)')
    ax3.set_title('轨道切线角度')
    ax3.grid(True, alpha=0.3)

    # 4. 曲率分布
    ax4 = axes[1, 0]
    ax4.plot(y_values[1:], curvature[1:], 'purple', linewidth=2)
    ax4.axvline(x=env.y_curve_start, color='orange', linestyle='--', alpha=0.5)
    ax4.axvline(x=env.y_curve_end, color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('纵向位置 y (m)')
    ax4.set_ylabel('曲率')
    ax4.set_title('轨道曲率分布')
    ax4.grid(True, alpha=0.3)

    # 5. 轨道参数汇总表
    ax5 = axes[1, 1]
    ax5.axis('off')

    # 计算关键点信息
    key_points = [
        ('机库位置', 0, env.track_centerline(0)),
        ('弯道起点', env.y_curve_start, env.track_centerline(env.y_curve_start)),
        ('弯道中点', (env.y_curve_start + env.y_curve_end)/2,
         env.track_centerline((env.y_curve_start + env.y_curve_end)/2)),
        ('弯道终点', env.y_curve_end, env.track_centerline(env.y_curve_end)),
        ('起点', env.y_start, env.track_centerline(env.y_start)),
    ]

    table_data = [['位置', 'y (m)', 'x (m)', '切线角 (度)']]
    for name, y, x in key_points:
        angle_deg = np.rad2deg(env.track_angle(y))
        table_data.append([name, f'{y:.2f}', f'{x:.3f}', f'{angle_deg:.1f}'])

    # 创建表格
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    table[(i, j)].set_facecolor('#e6e6e6')
                else:
                    table[(i, j)].set_facecolor('#f5f5f5')

    ax5.set_title('轨道关键点信息', fontsize=12, fontweight='bold')

    # 6. 速度建议
    ax6 = axes[1, 2]
    ax6.axis('off')

    # 根据曲率计算建议速度
    y_safe = np.linspace(0, env.y_start, 100)
    safe_speed = []
    for y in y_safe:
        curv = abs(env.track_derivative(y))
        if y < env.y_curve_start or y > env.y_curve_end:
            # 直线段
            speed = 0.025
        else:
            # 弯道：速度与曲率成反比
            speed = 0.012 * (1 - curv * 0.5)
            speed = np.clip(speed, 0.008, 0.018)
        safe_speed.append(speed)

    ax6.plot(y_safe, safe_speed, 'b-', linewidth=2)
    ax6.fill_between(y_safe, safe_speed, 0, alpha=0.3)
    ax6.axvline(x=env.y_curve_start, color='orange', linestyle='--', alpha=0.5)
    ax6.axvline(x=env.y_curve_end, color='orange', linestyle='--', alpha=0.5)
    ax6.set_xlabel('纵向位置 y (m)')
    ax6.set_ylabel('建议速度 (m/s)')
    ax6.set_title('建议牵引速度策略')
    ax6.grid(True, alpha=0.3)

    # 添加建议速度区间的文本
    ax6.text(env.y_curve_start + 0.1, 0.022, '高速区', fontsize=9, ha='center')
    ax6.text((env.y_curve_start + env.y_curve_end)/2, 0.01, '弯道减速区', fontsize=9, ha='center')
    ax6.text(env.y_curve_end + 0.1, 0.022, '高速区', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig('track_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印详细分析报告
    print("\n" + "="*60)
    print("轨道分析报告")
    print("="*60)
    print(f"\n轨道参数:")
    print(f"  起点位置: y = {env.y_start} m, x = {env.track_centerline(env.y_start):.3f} m")
    print(f"  终点位置: y = {env.y_end} m, x = {env.track_centerline(env.y_end):.3f} m")
    print(f"  弯道范围: y = [{env.y_curve_start}, {env.y_curve_end}] m")
    print(f"  横向偏移: {env.curve_dx} m")

    print(f"\n轨道特性:")
    max_deriv = max(abs(d) for d in derivative)
    max_angle = max(abs(a) for a in angle)
    print(f"  最大导数 |dx/dy|: {max_deriv:.3f}")
    print(f"  最大切线角度: {max_angle:.1f}°")

    print(f"\n控制建议:")
    print(f"  弯道前速度: 0.015 - 0.020 m/s")
    print(f"  弯道内速度: 0.008 - 0.012 m/s")
    print(f"  弯道后速度: 0.015 - 0.025 m/s")
    print(f"  横向速度限制: {env.VX_MAX} m/s (可提高到 0.006-0.008)")

    print(f"\n成功条件:")
    print(f"  允许偏差: ±{env.e_fm_limit} m")
    print(f"  允许偏角: ±{np.rad2deg(env.theta_limit):.0f}°")

    return {
        'y_values': y_values,
        'centerline': centerline,
        'derivative': derivative,
        'angle': angle,
        'curvature': curvature
    }

def plot_sample_trajectory():
    """绘制示例轨迹"""
    env = HelicopterInboundKinematicsEnv(fast_mode=False)

    # 模拟一个成功的轨迹（理想情况）
    y_traj = np.linspace(env.y_start, 0, 200)
    x_ideal = [env.track_centerline(y) for y in y_traj]

    # 模拟带偏差的轨迹
    np.random.seed(42)
    x_noisy = [x_ideal[i] + np.random.normal(0, 0.02) for i in range(len(y_traj))]

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制轨道区域
    ax.fill_between(y_traj,
                    [x - 0.1 for x in x_ideal],
                    [x + 0.1 for x in x_ideal],
                    color='blue', alpha=0.2, label='允许偏差范围 (±0.1m)')

    # 绘制理想轨迹
    ax.plot(y_traj, x_ideal, 'b-', linewidth=2, label='理想轨迹')

    # 绘制带偏差的轨迹
    ax.plot(y_traj, x_noisy, 'r--', linewidth=1.5, alpha=0.7, label='带偏差轨迹示例')

    # 标记弯道区域
    ax.axvspan(env.y_curve_start, env.y_curve_end, alpha=0.3, color='yellow', label='弯道区域')

    # 标记起点和终点
    ax.scatter([env.y_start], [env.track_centerline(env.y_start)],
               c='green', s=100, marker='s', label='起点 (降落区域)', zorder=5)
    ax.scatter([env.y_end], [env.track_centerline(env.y_end)],
               c='red', s=100, marker='s', label='终点 (机库)', zorder=5)

    # 添加方向箭头
    ax.annotate('牵引方向', xy=(2, 0.2), xytext=(3, 0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlabel('纵向位置 y (m)', fontsize=12)
    ax.set_ylabel('横向位置 x (m)', fontsize=12)
    ax.set_title('直升机入库轨迹示例', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, env.y_start + 0.2)
    ax.set_ylim(-0.3, 0.7)

    plt.tight_layout()
    plt.savefig('sample_trajectory.png', dpi=150)
    plt.show()

def analyze_control_requirements():
    """分析控制需求"""
    env = HelicopterInboundKinematicsEnv(fast_mode=False)

    print("\n" + "="*60)
    print("控制需求分析")
    print("="*60)

    # 计算弯道所需的最大角速度
    y_curve = np.linspace(env.y_curve_start, env.y_curve_end, 100)
    max_omega_needed = 0
    max_vx_needed = 0

    for y in y_curve:
        # 轨道切线角速度
        dtheta_dy = env.track_derivative(y)  # 近似
        # 假设速度 vy = 0.015 m/s
        vy_assumed = 0.015
        omega_needed = dtheta_dy * vy_assumed
        vx_needed = omega_needed * env.L_PFM

        max_omega_needed = max(max_omega_needed, abs(omega_needed))
        max_vx_needed = max(max_vx_needed, abs(vx_needed))

    print(f"\n弯道控制需求 (vy=0.015 m/s):")
    print(f"  所需最大角速度: {max_omega_needed:.3f} rad/s")
    print(f"  所需最大横向速度: {max_vx_needed:.4f} m/s")
    print(f"  当前横向速度限制: {env.VX_MAX} m/s")

    if max_vx_needed > env.VX_MAX:
        print(f"  ⚠️ 警告: 所需横向速度超过限制!")
        print(f"  建议: 提高 VX_MAX 到 {max_vx_needed * 1.2:.4f} m/s 或降低弯道速度")
    else:
        print(f"  ✓ 横向速度足够")

    # 计算合适的弯道速度
    safe_vy_curve = env.VX_MAX * 0.8 / max_vx_needed * 0.015 if max_vx_needed > 0 else 0.015
    print(f"\n建议弯道速度:")
    print(f"  基于横向速度限制: {safe_vy_curve:.3f} m/s")
    print(f"  建议范围: 0.008 - 0.012 m/s")

if __name__ == "__main__":
    # 运行完整分析
    print("正在生成轨道分析图...")
    track_data = analyze_track()

    print("\n正在生成示例轨迹图...")
    plot_sample_trajectory()

    print("\n正在分析控制需求...")
    analyze_control_requirements()

    print("\n" + "="*60)
    print("所有图表已保存:")
    print("  - track_analysis.png (轨道分析图)")
    print("  - sample_trajectory.png (示例轨迹图)")
    print("="*60)