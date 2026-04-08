"""
直升机自动牵引入库环境
参数：总长3.55m，最大速度0.1m/s
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class HelicopterInboundKinematicsEnv(gym.Env):
    """
    直升机自动牵引入库环境
    坐标系：y轴正方向指向机库（向前），x轴正方向向右
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, easy_mode=False):
        super().__init__()

        # ========== 几何参数（修正为合理值） ==========
        self.L_PFM = 0.198  # 牵引杆P到前轮中点FM的距离 (m) - 恢复原值
        self.L_AFM = 1.2  # 尾轮A到前轮中点FM的距离 (m) - 恢复原值
        self.L_PA = self.L_AFM - self.L_PFM

        # ========== 控制参数 ==========
        self.VX_MAX = 0.004  # 最大横移速度 (m/s)
        self.VY_MAX = 0.03  # 最大牵引速度 (m/s)
        self.VY_MIN = 0.005  # 最小牵引速度 (m/s)

        # ========== 轨道参数 ==========
        self.y_start = 0.0  # 起始y坐标（机库位置）
        self.y_end = 3.55  # 终点y坐标（降落区域）
        self.y_curve_start = 1.4  # 弯道起点
        self.y_curve_end = 2.65  # 弯道终点
        self.curve_dx = 0.38  # 横向偏移 (m)

        # ========== 控制阈值 ==========
        self.e_fm_limit = 0.1  # 允许偏差 (m)
        self.theta_limit = np.deg2rad(12)  # 允许偏角12度
        self.tail_angle_limit = np.deg2rad(10)

        # ========== 动作空间 ==========
        self.action_space = spaces.Box(
            low=np.array([-1, self.VY_MIN], dtype=np.float32),
            high=np.array([1, self.VY_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # ========== 观测空间 ==========
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -np.pi / 4, -0.5, -np.pi / 2, 0], dtype=np.float32),
            high=np.array([0.5, np.pi / 4, 0.5, np.pi / 2, 4], dtype=np.float32),
            dtype=np.float32
        )

        # 状态变量
        self.x_fm = None
        self.y_fm = None
        self.theta = None
        self.tail_angle = None
        self.x_p = None
        self.y_p = None

        self.current_steps = 0
        self.max_steps = 1000  # 足够完成全程
        self.trajectory = []

        self.last_e_fm = None
        self.last_theta_rel = None
        self.last_y = None

        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.easy_mode = easy_mode

    def track_centerline(self, y):
        """轨道中心线函数"""
        if y <= self.y_curve_start:
            return 0.0
        elif y <= self.y_curve_end:
            t = (y - self.y_curve_start) / (self.y_curve_end - self.y_curve_start)
            return self.curve_dx * (1 - np.cos(np.pi * t)) / 2
        else:
            return self.curve_dx

    def track_derivative(self, y):
        """轨道中心线导数 dx/dy"""
        if y <= self.y_curve_start:
            return 0.0
        elif y <= self.y_curve_end:
            t = (y - self.y_curve_start) / (self.y_curve_end - self.y_curve_start)
            dt_dy = 1 / (self.y_curve_end - self.y_curve_start)
            dx_dt = self.curve_dx * np.pi * np.sin(np.pi * t) / 2
            return dx_dt * dt_dy
        else:
            return 0.0

    def track_angle(self, y):
        """轨道切线角度 (rad)"""
        return np.arctan2(self.track_derivative(y), 1.0)

    def update_kinematics(self, vx_cmd, vy_cmd, dt):
        """
        运动学更新（修正版）
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 牵引杆速度（世界坐标系）
        v_px = vx_cmd * cos_theta - vy_cmd * sin_theta
        v_py = vx_cmd * sin_theta + vy_cmd * cos_theta

        # 角速度（使用更稳定的公式）
        # omega = (vx_cmd * cos_theta + vy_cmd * sin_theta) / L_PFM
        # 简化：横向速度引起的旋转
        omega = vx_cmd / self.L_PFM

        # 限制角速度范围
        omega = np.clip(omega, -0.5, 0.5)

        # 前轮中点速度
        v_fm_x = v_px - omega * self.L_PFM * cos_theta
        v_fm_y = v_py - omega * self.L_PFM * sin_theta

        # 更新状态
        self.x_fm += v_fm_x * dt
        self.y_fm += v_fm_y * dt
        self.theta += omega * dt

        # 限制位置范围（防止飞出）
        self.x_fm = np.clip(self.x_fm, -0.5, 1.0)
        self.y_fm = np.clip(self.y_fm, -0.5, self.y_end + 0.5)
        self.theta = np.clip(self.theta, -np.pi / 3, np.pi / 3)

        # 更新牵引杆位置
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 尾轮偏角（简化模型）
        self.tail_angle = np.clip(self.tail_angle + omega * dt * 0.5, -0.3, 0.3)

    def compute_reward(self, e_fm, theta_rel, y_fm):
        """
        激进版奖励函数 - 强调精度
        """
        reward = 0.0

        # ========== 1. 精度惩罚（指数级） ==========
        # 偏差惩罚 - 指数增长
        e_penalty = 10.0 * (abs(e_fm) ** 1.5)  # 0.1m→0.32, 0.2m→0.89, 0.3m→1.64
        theta_penalty = 8.0 * (abs(theta_rel) ** 1.5)
        reward -= e_penalty
        reward -= theta_penalty

        # ========== 2. 精度奖励（阶梯式） ==========
        # 偏差奖励
        if abs(e_fm) < 0.03:
            reward += 3.0  # 优秀
        elif abs(e_fm) < 0.05:
            reward += 1.5  # 良好
        elif abs(e_fm) < 0.08:
            reward += 0.5  # 及格

        # 偏角奖励
        if abs(theta_rel) < 0.02:  # ~1.15°
            reward += 2.0
        elif abs(theta_rel) < 0.05:  # ~2.86°
            reward += 1.0
        elif abs(theta_rel) < 0.08:  # ~4.58°
            reward += 0.3

        # ========== 3. 改善奖励（增强） ==========
        if self.last_e_fm is not None:
            e_improve = abs(self.last_e_fm) - abs(e_fm)
            theta_improve = abs(self.last_theta_rel) - abs(theta_rel)

            # 改善越多，奖励越大
            if e_improve > 0:
                reward += 3.0 * e_improve * (1 + abs(e_fm) * 10)  # 偏差大时改善奖励更高
            elif e_improve < -0.02:  # 恶化惩罚
                reward -= 1.0

            if theta_improve > 0:
                reward += 2.0 * theta_improve * (1 + abs(theta_rel) * 20)

        # ========== 4. 条件前进奖励 ==========
        # 只有精度足够时才给前进奖励
        if abs(e_fm) < 0.1 and abs(theta_rel) < 0.1:
            # 良好状态下的前进奖励
            progress = (self.y_end - y_fm) / self.y_end
            reward += 2.0 * progress
        else:
            # 状态差时，前进奖励很少
            progress = (self.y_end - y_fm) / self.y_end
            reward += 0.2 * progress

        # ========== 5. 到达奖励（严格） ==========
        if y_fm >= self.y_end:
            if abs(e_fm) < 0.03 and abs(theta_rel) < 0.02:
                reward += 300  # 完美入库
                print(f"🏆 完美入库！偏差={e_fm:.3f}m, 偏角={np.rad2deg(theta_rel):.1f}°")
            elif abs(e_fm) < 0.05 and abs(theta_rel) < 0.05:
                reward += 150  # 优秀入库
                print(f"⭐ 优秀入库！偏差={e_fm:.3f}m, 偏角={np.rad2deg(theta_rel):.1f}°")
            elif abs(e_fm) < 0.08 and abs(theta_rel) < 0.08:
                reward += 80  # 良好入库
                print(f"✓ 良好入库！偏差={e_fm:.3f}m, 偏角={np.rad2deg(theta_rel):.1f}°")
            elif abs(e_fm) < 0.1 and abs(theta_rel) < 0.1:
                reward += 30  # 及格
                print(f"⚠️ 及格入库！偏差={e_fm:.3f}m, 偏角={np.rad2deg(theta_rel):.1f}°")
            else:
                reward -= 50  # 失败
                print(f"✗ 到达但位姿不佳 | 偏差: {e_fm:.3f}m, 偏角: {np.rad2deg(theta_rel):.1f}°")

        # ========== 6. 稳定性奖励 ==========
        reward += 0.02

        # ========== 7. 大幅度动作惩罚 ==========
        if hasattr(self, 'last_vx_cmd'):
            if abs(self.last_vx_cmd) > 0.08:
                reward -= 0.2

        return reward

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        if self.easy_mode:
            # 简单模式：从靠近机库的位置开始
            self.y_fm = self.np_random.uniform(0, 1.5)
            e_fm_init = self.np_random.uniform(-0.1, 0.1)
            self.theta = self.np_random.uniform(-np.deg2rad(5), np.deg2rad(5))
        else:
            # 正常模式：从起点附近开始
            self.y_fm = self.np_random.uniform(2.8, self.y_end)
            e_fm_init = self.np_random.uniform(-0.25, 0.25)
            self.theta = self.np_random.uniform(-np.deg2rad(12), np.deg2rad(12))

        track_x = self.track_centerline(self.y_fm)
        self.x_fm = track_x + e_fm_init

        self.tail_angle = 0.0
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        self.current_steps = 0
        self.trajectory = []

        obs = self._get_obs()
        self.last_e_fm = obs[0]
        self.last_theta_rel = obs[1]
        self.last_y = self.y_fm

        self._record_state()

        return obs, {}

    def _get_obs(self):
        """获取观测值"""
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        # 理想牵引杆位置
        ideal_x_p = self.track_centerline(self.y_fm) - self.L_PFM * np.sin(self.track_angle(self.y_fm))
        e_p = self.x_p - ideal_x_p

        y_remaining = self.y_end - self.y_fm

        return np.array([
            e_fm,
            theta_rel,
            e_p,
            self.tail_angle,
            y_remaining
        ], dtype=np.float32)

    def _record_state(self):
        """记录状态"""
        self.trajectory.append({
            'x_fm': self.x_fm,
            'y_fm': self.y_fm,
            'theta': self.theta,
            'x_p': self.x_p,
            'y_p': self.y_p,
            'tail_angle': self.tail_angle
        })

    def step(self, action):
        """执行一步"""
        dt = 0.05

        vx_sign = np.clip(action[0], -1, 1)
        vy = np.clip(action[1], self.VY_MIN, self.VY_MAX)

        vx_cmd = vx_sign * self.VX_MAX
        # 记录动作用于惩罚
        self.last_vx_cmd = vx_cmd

        # 更新动力学
        self.update_kinematics(vx_cmd, vy, dt)
        self._record_state()

        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        reward = self.compute_reward(e_fm, theta_rel, self.y_fm)

        self.current_steps += 1
        self.last_e_fm = e_fm
        self.last_theta_rel = theta_rel

        # 终止条件（修正：到达y_end才算成功）
        terminated = self.y_fm >= self.y_end

        # 失败条件（放宽一点）
        failed = (abs(e_fm) > 0.5 or  # 偏差大于0.5m
                  abs(theta_rel) > np.deg2rad(45) or  # 偏角大于45度
                  self.current_steps >= self.max_steps)

        truncated = failed

        # 成功奖励
        if terminated:
            if abs(e_fm) < self.e_fm_limit and abs(theta_rel) < self.theta_limit:
                reward += 100
                print(
                    f"✓ 成功入库！步数: {self.current_steps}, 最终偏差: {e_fm:.3f}m, 偏角: {np.rad2deg(theta_rel):.1f}°")
            else:
                reward -= 30
                print(f"✗ 到达但位姿不佳 | 偏差: {e_fm:.3f}m, 偏角: {np.rad2deg(theta_rel):.1f}°")

        if failed and not terminated:
            reward -= 15

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """渲染"""
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))

        self.ax.clear()

        # 设置坐标轴
        self.ax.set_xlim(-0.2, 0.6)
        self.ax.set_ylim(-0.5, self.y_end + 0.5)
        self.ax.set_xlabel("横向位置 x (m)", fontsize=12)
        self.ax.set_ylabel("纵向位置 y (m) → 机库方向", fontsize=12)
        self.ax.set_title("直升机自动牵引入库仿真", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # 绘制轨道
        ys = np.linspace(-0.5, self.y_end + 0.5, 500)
        xs = [self.track_centerline(y) for y in ys]
        self.ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道', zorder=1)

        # 轨道边界
        self.ax.fill_between(ys,
                             [x - 0.05 for x in xs],
                             [x + 0.05 for x in xs],
                             color='blue', alpha=0.1, zorder=0)

        # 机库区域
        hangar_rect = Rectangle((-0.3, -0.3), 0.6, 0.5,
                                linewidth=2, edgecolor='red',
                                facecolor='none', label='机库')
        self.ax.add_patch(hangar_rect)

        # 绘制直升机
        self._draw_helicopter()

        # 绘制轨迹
        if len(self.trajectory) > 1:
            fm_traj_x = [t['x_fm'] for t in self.trajectory[::5]]
            fm_traj_y = [t['y_fm'] for t in self.trajectory[::5]]
            self.ax.plot(fm_traj_x, fm_traj_y, 'g--', linewidth=1,
                         alpha=0.5, label='前轮轨迹', zorder=1)

        # 显示信息
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        info_text = f"步数: {self.current_steps}\n"
        info_text += f"位置: {self.y_fm:.2f}/{self.y_end:.2f}m\n"
        info_text += f"进度: {self.y_fm / self.y_end * 100:.1f}%\n"
        info_text += f"偏差: {e_fm:.3f}m\n"
        info_text += f"偏角: {np.rad2deg(theta_rel):.1f}°"

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.ax.legend(loc='upper right')
        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.01)

    def _draw_helicopter(self):
        """绘制直升机"""
        fm_x, fm_y = self.x_fm, self.y_fm

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 机头
        nose = (fm_x, fm_y)

        # 机尾
        tail_x = fm_x - self.L_AFM * sin_theta
        tail_y = fm_y - self.L_AFM * cos_theta

        # 机身
        left_rear = (tail_x - 0.08 * cos_theta, tail_y + 0.08 * sin_theta)
        right_rear = (tail_x + 0.08 * cos_theta, tail_y - 0.08 * sin_theta)

        body = Polygon([nose, left_rear, right_rear],
                       closed=True, color='gray', alpha=0.8, zorder=2)
        self.ax.add_patch(body)

        # 前轮
        front_wheel = Circle(nose, 0.04, color='black', zorder=3)
        self.ax.add_patch(front_wheel)

        # 尾轮
        tail_wheel = Circle((tail_x, tail_y), 0.035, color='red', zorder=3)
        self.ax.add_patch(tail_wheel)

        # 牵引杆
        towbar_x = self.x_p
        towbar_y = self.y_p
        self.ax.plot([towbar_x, fm_x], [towbar_y, fm_y],
                     linewidth=2, color='orange', zorder=2)
        self.ax.plot(towbar_x, towbar_y, 'ro', markersize=4, zorder=3)

    def make_animation(self, filename="helicopter_inbound.gif", save_frames=200):
        """生成动画"""
        if len(self.trajectory) == 0:
            print("无轨迹数据")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        def animate(frame_idx):
            ax.clear()

            t = self.trajectory[min(frame_idx, len(self.trajectory) - 1)]
            x_fm, y_fm = t['x_fm'], t['y_fm']
            theta = t['theta']
            x_p, y_p = t['x_p'], t['y_p']

            ax.set_xlim(-0.2, 0.6)
            ax.set_ylim(-0.5, self.y_end + 0.5)
            ax.set_xlabel("横向位置 x (m)")
            ax.set_ylabel("纵向位置 y (m)")
            ax.set_title(f"直升机自动牵引入库 (步数: {frame_idx})")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            ys = np.linspace(-0.5, self.y_end + 0.5, 500)
            xs = [self.track_centerline(y) for y in ys]
            ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道')

            hangar_rect = Rectangle((-0.3, -0.3), 0.6, 0.5,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none', label='机库')
            ax.add_patch(hangar_rect)

            traj_x = [t_hist['x_fm'] for t_hist in self.trajectory[:frame_idx + 1:5]]
            traj_y = [t_hist['y_fm'] for t_hist in self.trajectory[:frame_idx + 1:5]]
            ax.plot(traj_x, traj_y, 'g--', linewidth=1, alpha=0.5, label='前轮轨迹')

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            nose = (x_fm, y_fm)
            tail_x = x_fm - self.L_AFM * sin_theta
            tail_y = y_fm - self.L_AFM * cos_theta
            left_rear = (tail_x - 0.08 * cos_theta, tail_y + 0.08 * sin_theta)
            right_rear = (tail_x + 0.08 * cos_theta, tail_y - 0.08 * sin_theta)

            body = Polygon([nose, left_rear, right_rear],
                           closed=True, color='gray', alpha=0.8)
            ax.add_patch(body)

            ax.plot([x_p, x_fm], [y_p, y_fm], 'orange', linewidth=2)
            ax.plot(x_p, y_p, 'ro', markersize=4)

            e_fm = x_fm - self.track_centerline(y_fm)
            info_text = f"偏差: {e_fm:.3f}m | y: {y_fm:.2f}m"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.legend(loc='upper right')
            return []

        step = max(1, len(self.trajectory) // save_frames)
        frames = list(range(0, len(self.trajectory), step))

        ani = FuncAnimation(fig, animate, frames=frames,
                            interval=50, blit=True, repeat=False)
        ani.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        print(f"动画已保存至: {filename}")

    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ========== 测试 ==========
if __name__ == "__main__":
    env = HelicopterInboundKinematicsEnv(render_mode="human")

    print("=" * 50)
    print("直升机入库环境测试")
    print(f"轨道长度: {env.y_end}m")
    print(f"弯道范围: {env.y_curve_start}-{env.y_curve_end}m")
    print(f"横向偏移: {env.curve_dx}m")
    print(f"最大速度: {env.VY_MAX}m/s")
    print("=" * 50)

    success_count = 0

    for episode in range(10):
        obs, _ = env.reset()
        print(f"\nEpisode {episode + 1} - 起始位置 y={obs[4]:.2f}m")

        total_reward = 0

        for step in range(500):
            e_fm, theta_rel, e_p, tail_angle, y_remaining = obs

            # PID控制
            vx_sign = -np.clip(e_fm * 4.0, -1, 1)  # 更强修正
            vy = 0.08  # 中等速度

            action = np.array([vx_sign, vy])
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                success = terminated and abs(e_fm) < 0.1
                if success:
                    success_count += 1
                status = "✓ 成功" if success else "✗ 失败"
                print(f"{status} | 步数: {step + 1} | y={obs[4]:.2f}m | 偏差={obs[0]:.3f}m | 总奖励={total_reward:.1f}")
                break

        if not (terminated or truncated):
            print(f"超时 | y={obs[4]:.2f}m | 偏差={obs[0]:.3f}m")

    print(f"\n{'=' * 50}")
    print(f"测试完成！成功率: {success_count}/10 = {success_count * 10}%")
    print(f"{'=' * 50}")

    env.close()