"""
直升机自动牵引入库环境
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.animation import FuncAnimation

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class HelicopterInboundKinematicsEnv(gym.Env):
    """
    直升机自动牵引入库环境
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # ========== 几何参数（基于论文图16） ==========
        self.L_PFM = 0.198  # 牵引杆P到前轮中点FM的距离 (m)
        self.L_AFM = 1.2  # 尾轮A到前轮中点FM的距离 (m)
        self.L_PA = self.L_AFM - self.L_PFM  # 牵引杆P到尾轮A的距离 (m) = 1.002
        self.L_wheel = 0.3  # 尾轮轮轴到触地点距离 (m)

        # ========== 控制参数 ==========
        # 根据论文第4页指标：横移速度0.05m/s恒定，牵引速度0.05-0.5m/s可调
        self.VX_CONSTANT = 0.05  # 横移速度恒定值 (m/s)
        self.VY_MIN = 0.05  # 最小牵引速度 (m/s)
        self.VY_MAX = 0.5  # 最大牵引速度 (m/s)

        # ========== 轨道参数 ==========
        self.y_start = 30.0  # 起始y坐标 (降落区域)
        self.y_curve_start = 22.0  # 弯道起点
        self.y_curve_end = 8.0  # 弯道终点
        self.y_hangar_end = 0.0  # 机库终点
        self.curve_dx = 0.6  # 弯道横向偏移 (m)

        # ========== 控制阈值（论文3.2节） ==========
        self.e_fm_limit = 0.3  # 前轮中点允许偏差 (m)
        self.theta_limit = np.deg2rad(12)  # 允许偏角上限 (12°)
        self.tail_angle_limit = np.deg2rad(8)  # 尾轮偏角限制 (8°)

        # ========== 动作空间 ==========
        # 动作: [vx_sign, vy]
        # vx_sign: -1(左移), 0(不动), 1(右移)  - 横移速度方向
        # vy: 牵引速度大小 (连续值)
        self.action_space = spaces.Box(
            low=np.array([-1, self.VY_MIN], dtype=np.float32),
            high=np.array([1, self.VY_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # ========== 观测空间 ==========
        # [e_fm, theta, e_p, tail_angle, y_remaining]
        # e_fm: 前轮中点距轨道距离 (m)
        # theta: 机身偏角 (rad)
        # e_p: 牵引杆距理想轨迹距离 (m)
        # tail_angle: 尾轮偏角 (rad)
        # y_remaining: 剩余入库距离 (m)
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -np.pi / 6, -0.5, -np.pi / 3, 0], dtype=np.float32),
            high=np.array([0.5, np.pi / 6, 0.5, np.pi / 3, 35], dtype=np.float32),
            dtype=np.float32
        )

        # ========== 状态变量 ==========
        self.x_fm = None  # 前轮中点FM的x坐标 (m)
        self.y_fm = None  # 前轮中点FM的y坐标 (m)
        self.theta = None  # 机身偏角 (rad)
        self.tail_angle = None  # 尾轮偏角 (rad) - 尾轮朝向与机身的夹角
        self.x_p = None  # 牵引杆P的x坐标 (m)
        self.y_p = None  # 牵引杆P的y坐标 (m)

        self.current_steps = 0
        self.max_steps = 1500
        self.trajectory = []  # 记录轨迹用于可视化

        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        # 用于差分计算的上一帧位置
        self._prev_x_A = None
        self._prev_y_A = None

    def track_centerline(self, y):
        """
        轨道中心线函数（图87：直-弯-直线型）

        参数:
            y: 沿轨道位置 (m)
        返回:
            x: 轨道中心线x坐标 (m)
        """
        if y >= self.y_curve_start:
            # 直轨道段1
            return 0.0
        elif y >= self.y_curve_end:
            # 弯轨道段（贝塞尔曲线过渡）
            t = (self.y_curve_start - y) / (self.y_curve_start - self.y_curve_end)
            # 使用S形曲线，曲率连续
            return self.curve_dx * (1 - np.cos(np.pi * t)) / 2
        else:
            # 直轨道段2
            return self.curve_dx

    def track_derivative(self, y):
        """
        轨道中心线导数 dx/dy
        """
        if y >= self.y_curve_start:
            return 0.0
        elif y >= self.y_curve_end:
            t = (self.y_curve_start - y) / (self.y_curve_start - self.y_curve_end)
            dt_dy = -1 / (self.y_curve_start - self.y_curve_end)
            dx_dt = self.curve_dx * np.pi * np.sin(np.pi * t) / 2
            return dx_dt * dt_dy
        else:
            return 0.0

    def track_angle(self, y):
        """
        轨道切线角度 (rad)
        """
        return np.arctan2(self.track_derivative(y), 1.0)

    def ideal_trajectory(self, y):
        """
        理想轨迹
        假设前轮中点始终在轨道上且机身与轨道相切
        返回: 理想牵引杆位置 x_p_ideal
        """
        # 前轮中点位置（在轨道上）
        x_fm_ideal = self.track_centerline(y)
        # 前轮中点处轨道角度
        track_angle = self.track_angle(y)
        # 理想牵引杆位置
        x_p_ideal = x_fm_ideal - self.L_PFM * np.sin(track_angle)
        return x_p_ideal

    def update_kinematics(self, vx, vy, dt):
        """
        参数:
            vx: 牵引杆横移速度 (m/s) - 垂直轨道方向
            vy: 牵引杆牵引速度 (m/s) - 沿轨道方向
            dt: 时间步长 (s)
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 前轮中点速度
        # v_FM = v_P + ω × r_PFM
        vx_fm = vx * cos_theta - vy * sin_theta
        vy_fm = -vy * cos_theta

        # 机身角速度
        # ω = (vx·cosθ + vy·sinθ) / L_PFM
        omega = (vx * cos_theta + vy * sin_theta) / self.L_PFM

        # 更新状态
        self.x_fm += vx_fm * dt
        self.y_fm += vy_fm * dt
        self.theta += omega * dt

        # 角度归一化到[-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # 更新牵引杆位置（式2.1）
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 更新尾轮偏角（式2.9-2.12）
        self._update_tail_angle(dt, omega)

    def _update_tail_angle(self, dt, omega):
        """
        更新尾轮偏角
        尾轮作为万向轮，其朝向会自然趋向于运动方向

        参数:
            dt: 时间步长
            omega: 机身角速度
        """
        # 计算尾轮A点位置
        x_A = self.x_fm - self.L_AFM * np.sin(self.theta)
        y_A = self.y_fm - self.L_AFM * np.cos(self.theta)

        # 使用差分计算A点速度（需要至少两帧数据）
        if self._prev_x_A is not None and self._prev_y_A is not None and dt > 0:
            v_Ax = (x_A - self._prev_x_A) / dt
            v_Ay = (y_A - self._prev_y_A) / dt
        else:
            # 第一帧：使用解析方法估算速度
            # 基于当前角速度和牵引杆速度估算
            v_Ax = self.x_p + self.L_PA * np.cos(self.theta) * omega
            v_Ay = self.y_p + self.L_PA * np.sin(self.theta) * omega

        # 保存当前位置用于下次差分
        self._prev_x_A = x_A
        self._prev_y_A = y_A

        # A点速度方向
        v_A = np.sqrt(v_Ax ** 2 + v_Ay ** 2)

        if v_A > 0.01:
            # 速度方向角（世界坐标系）
            velocity_angle = np.arctan2(v_Ax, v_Ay)
            # 转换为相对于机身的角度
            target_angle = velocity_angle - self.theta
            # 归一化到[-π, π]
            target_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))

            # 一阶惯性响应（模拟尾轮转向）
            tau = 0.5  # 时间常数，越大响应越慢
            alpha = min(1.0, dt / tau)
            self.tail_angle += alpha * (target_angle - self.tail_angle)
        else:
            # 速度很小时，尾轮缓慢回到中立位置
            alpha = min(1.0, dt / 1.0)  # 较慢的回归速度
            self.tail_angle += alpha * (0 - self.tail_angle)

        # 限制尾轮偏角范围
        self.tail_angle = np.clip(self.tail_angle, -np.pi / 2, np.pi / 2)

    def compute_reward(self, e_fm, theta_rel, e_p, tail_angle, y_fm):
        """
        计算奖励函数

        控制目标：
        1. 减小前轮中点偏差 e_fm
        2. 减小机身偏角 theta
        3. 保持尾轮偏角在安全范围内
        4. 快速入库
        """
        reward = 0.0

        # ========== 主要目标：减小偏差 ==========
        # 前轮中点偏差惩罚
        reward -= 3.0 * abs(e_fm)

        # 机身偏角惩罚
        reward -= 2.0 * abs(theta_rel)

        # 牵引杆偏离理想轨迹惩罚
        reward -= 0.5 * abs(e_p)

        # ========== 尾轮安全约束 ==========
        # 尾轮偏角超出安全范围惩罚
        if abs(tail_angle) > self.tail_angle_limit:
            reward -= 5.0 * (abs(tail_angle) - self.tail_angle_limit)

        # ========== 进度奖励 ==========
        # 向机库移动奖励
        progress = (self.y_start - max(0, y_fm)) / self.y_start
        reward += 1.5 * progress

        # 阶段性奖励：位姿良好
        if abs(e_fm) < self.e_fm_limit and abs(theta_rel) < self.theta_limit:
            reward += 0.8

        return reward

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 随机初始位姿
        self.y_fm = self.np_random.uniform(self.y_start - 3, self.y_start)
        track_x = self.track_centerline(self.y_fm)

        # 初始偏差
        e_fm_init = self.np_random.uniform(-0.4, 0.4)
        self.x_fm = track_x + e_fm_init

        # 初始偏角 (-15° 到 15°)
        self.theta = self.np_random.uniform(-np.deg2rad(15), np.deg2rad(15))

        # 初始尾轮偏角（假设与机身对齐）
        self.tail_angle = 0.0

        # 计算初始牵引杆位置
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 重置差分变量
        self._prev_x_A = None
        self._prev_y_A = None

        self.current_steps = 0
        self.trajectory = []

        # 记录初始状态
        self._record_state()

        return self._get_obs(), {}

    def _get_obs(self):
        """获取观测值"""
        # 计算偏差
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        e_p = self.x_p - self.ideal_trajectory(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)
        y_remaining = max(0, self.y_fm - self.y_hangar_end)

        return np.array([
            e_fm,
            theta_rel,
            e_p,
            self.tail_angle,
            y_remaining
        ], dtype=np.float32)

    def _record_state(self):
        """记录当前状态用于可视化"""
        self.trajectory.append({
            'x_fm': self.x_fm,
            'y_fm': self.y_fm,
            'theta': self.theta,
            'x_p': self.x_p,
            'y_p': self.y_p,
            'tail_angle': self.tail_angle
        })

    def step(self, action):
        """
        执行一步仿真

        参数:
            action: [vx_sign, vy]
                vx_sign: -1, 0, 1  (横移速度方向)
                vy: 0.05-0.5 (牵引速度大小)
        """
        dt = 0.1  # 固定时间步长

        # 解析动作
        vx_sign = np.clip(action[0], -1, 1)
        vy = np.clip(action[1], self.VY_MIN, self.VY_MAX)

        # 横移速度 = 方向 × 恒定大小
        vx = vx_sign * self.VX_CONSTANT

        # 运动学更新
        self.update_kinematics(vx, vy, dt)

        # 记录状态
        self._record_state()

        # 计算奖励
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)
        e_p = self.x_p - self.ideal_trajectory(self.y_fm)
        reward = self.compute_reward(e_fm, theta_rel, e_p, self.tail_angle, self.y_fm)

        self.current_steps += 1

        # 终止条件
        terminated = self.y_fm <= self.y_hangar_end  # 到达机库
        truncated = self.current_steps >= self.max_steps

        # 失稳提前终止
        if abs(theta_rel) > np.deg2rad(30) or abs(e_fm) > 0.8:
            truncated = True
            reward -= 50

        # 成功额外奖励
        if terminated:
            if abs(e_fm) < self.e_fm_limit and abs(theta_rel) < self.theta_limit:
                reward += 200
            else:
                reward -= 50

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """渲染可视化"""
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))

        self.ax.clear()

        # 设置坐标轴
        self.ax.set_xlim(-0.3, 1.0)
        self.ax.set_ylim(-2, self.y_start + 2)
        self.ax.set_xlabel("横向位置 x (m)", fontsize=12)
        self.ax.set_ylabel("纵向位置 y (m)", fontsize=12)
        self.ax.set_title("直升机自动牵引入库仿真", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # 绘制轨道中心线
        ys = np.linspace(-1, self.y_start + 2, 500)
        xs = [self.track_centerline(y) for y in ys]
        self.ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道', zorder=1)

        # 绘制轨道边界（轨道宽度0.2m）
        self.ax.fill_between(ys,
                             [x - 0.1 for x in xs],
                             [x + 0.1 for x in xs],
                             color='blue', alpha=0.1, zorder=0)

        # 绘制机库区域
        hangar_x = self.track_centerline(0) - 0.5
        hangar_rect = Rectangle((hangar_x, -1), 1.0, 1.5,
                                linewidth=2, edgecolor='red',
                                facecolor='none', label='机库')
        self.ax.add_patch(hangar_rect)

        # 绘制直升机
        self._draw_helicopter()

        # 绘制轨迹
        if len(self.trajectory) > 1:
            fm_traj_x = [t['x_fm'] for t in self.trajectory]
            fm_traj_y = [t['y_fm'] for t in self.trajectory]
            self.ax.plot(fm_traj_x, fm_traj_y, 'g--', linewidth=1,
                         alpha=0.5, label='前轮中点轨迹', zorder=1)

        # 显示状态信息
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        info_text = f"步数: {self.current_steps}\n"
        info_text += f"偏差: {e_fm:.3f}m\n"
        info_text += f"偏角: {np.rad2deg(theta_rel):.1f}°\n"
        info_text += f"尾轮角: {np.rad2deg(self.tail_angle):.1f}°\n"
        info_text += f"y位置: {self.y_fm:.1f}m"

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.ax.legend(loc='upper right')

        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def _draw_helicopter(self):
        """绘制直升机（基于论文图16的坐标系）"""
        # 前轮中点FM位置
        fm_x, fm_y = self.x_fm, self.y_fm

        # 机身三角形顶点（后三点式布局）
        # 机头在前轮中点，机尾在后方L_AFM处
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 机头（前轮中点）
        nose = (fm_x, fm_y)

        # 左右后点（机尾两侧）
        tail_center_x = fm_x - self.L_AFM * sin_theta
        tail_center_y = fm_y - self.L_AFM * cos_theta

        left_rear = (tail_center_x - 0.1 * cos_theta,
                     tail_center_y + 0.1 * sin_theta)
        right_rear = (tail_center_x + 0.1 * cos_theta,
                      tail_center_y - 0.1 * sin_theta)

        # 绘制机身
        body = Polygon([nose, left_rear, right_rear],
                       closed=True, color='gray', alpha=0.8, zorder=2)
        self.ax.add_patch(body)

        # 绘制前轮（固定轮）
        front_wheel = Circle(nose, 0.08, color='black', zorder=3)
        self.ax.add_patch(front_wheel)

        # 绘制尾轮（万向轮）
        tail_wheel = Circle((tail_center_x, tail_center_y), 0.07,
                            color='red', zorder=3)
        self.ax.add_patch(tail_wheel)

        # 绘制尾轮朝向指示线（显示万向轮当前朝向）
        tail_wheel_angle = self.theta + self.tail_angle
        arrow_len = 0.15
        arrow_end_x = tail_center_x + arrow_len * np.sin(tail_wheel_angle)
        arrow_end_y = tail_center_y + arrow_len * np.cos(tail_wheel_angle)
        self.ax.annotate('', xy=(arrow_end_x, arrow_end_y),
                         xytext=(tail_center_x, tail_center_y),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # 绘制牵引杆
        towbar_x = self.x_p
        towbar_y = self.y_p
        self.ax.plot([towbar_x, fm_x], [towbar_y, fm_y],
                     linewidth=3, color='orange', zorder=2, label='牵引杆')

        # 标记牵引点
        self.ax.plot(towbar_x, towbar_y, 'ro', markersize=6, zorder=3)

        # 绘制理想轨迹上的对应点
        ideal_x = self.ideal_trajectory(self.y_fm)
        self.ax.plot(ideal_x, self.y_fm, 'g*', markersize=8,
                     alpha=0.5, label='理想牵引杆位置')

    def make_animation(self, filename="helicopter_inbound.gif", save_frames=500):
        """生成动画"""
        if len(self.trajectory) == 0:
            print("无轨迹数据")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        def animate(frame_idx):
            ax.clear()

            # 获取当前帧数据
            t = self.trajectory[min(frame_idx, len(self.trajectory) - 1)]
            x_fm, y_fm = t['x_fm'], t['y_fm']
            theta = t['theta']
            x_p, y_p = t['x_p'], t['y_p']
            tail_angle = t['tail_angle']

            # 设置坐标轴
            ax.set_xlim(-0.3, 1.0)
            ax.set_ylim(-2, self.y_start + 2)
            ax.set_xlabel("横向位置 x (m)")
            ax.set_ylabel("纵向位置 y (m)")
            ax.set_title(f"直升机自动牵引入库 (步数: {frame_idx})")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # 绘制轨道
            ys = np.linspace(-1, self.y_start + 2, 500)
            xs = [self.track_centerline(y) for y in ys]
            ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道')

            # 绘制机库
            hangar_x = self.track_centerline(0) - 0.5
            hangar_rect = Rectangle((hangar_x, -1), 1.0, 1.5,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none', label='机库')
            ax.add_patch(hangar_rect)

            # 绘制历史轨迹
            traj_x = [t_hist['x_fm'] for t_hist in self.trajectory[:frame_idx + 1]]
            traj_y = [t_hist['y_fm'] for t_hist in self.trajectory[:frame_idx + 1]]
            ax.plot(traj_x, traj_y, 'g--', linewidth=1, alpha=0.5, label='前轮轨迹')

            # 绘制直升机
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            nose = (x_fm, y_fm)
            tail_center_x = x_fm - self.L_AFM * sin_theta
            tail_center_y = y_fm - self.L_AFM * cos_theta
            left_rear = (tail_center_x - 0.1 * cos_theta,
                         tail_center_y + 0.1 * sin_theta)
            right_rear = (tail_center_x + 0.1 * cos_theta,
                          tail_center_y - 0.1 * sin_theta)

            body = Polygon([nose, left_rear, right_rear],
                           closed=True, color='gray', alpha=0.8)
            ax.add_patch(body)

            front_wheel = Circle(nose, 0.08, color='black')
            ax.add_patch(front_wheel)

            tail_wheel = Circle((tail_center_x, tail_center_y), 0.07, color='red')
            ax.add_patch(tail_wheel)

            # 尾轮朝向
            tail_wheel_angle = theta + tail_angle
            arrow_len = 0.15
            arrow_end_x = tail_center_x + arrow_len * np.sin(tail_wheel_angle)
            arrow_end_y = tail_center_y + arrow_len * np.cos(tail_wheel_angle)
            ax.annotate('', xy=(arrow_end_x, arrow_end_y),
                        xytext=(tail_center_x, tail_center_y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

            # 牵引杆
            ax.plot([x_p, x_fm], [y_p, y_fm], 'orange', linewidth=3)
            ax.plot(x_p, y_p, 'ro', markersize=6)

            # 理想位置
            ideal_x = self.ideal_trajectory(y_fm)
            ax.plot(ideal_x, y_fm, 'g*', markersize=8, alpha=0.5)

            # 状态信息
            e_fm = x_fm - self.track_centerline(y_fm)
            theta_rel = theta - self.track_angle(y_fm)
            info_text = f"偏差: {e_fm:.3f}m | 偏角: {np.rad2deg(theta_rel):.1f}° | y: {y_fm:.1f}m"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.legend(loc='upper right')
            return []

        # 采样帧（避免太多帧）
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


# ========== 测试代码 ==========
if __name__ == "__main__":
    env = HelicopterInboundKinematicsEnv(render_mode="human")

    obs, _ = env.reset()

    for _ in range(500):
        # 简单的试探性策略
        action = np.array([0, 0.15])  # 直行，不横移

        # 如果有偏差，尝试修正
        if obs[0] > 0.05:  # 前轮偏右
            action[0] = -1  # 向左横移
        elif obs[0] < -0.05:  # 前轮偏左
            action[0] = 1  # 向右横移

        obs, reward, terminated, truncated, _ = env.step(action)

        if env.render_mode == "human":
            env.render()

        if terminated or truncated:
            print(f"Episode结束: terminated={terminated}, truncated={truncated}")
            print(f"最终偏差: {obs[0]:.3f}m, 偏角: {np.rad2deg(obs[1]):.1f}°")
            break

    env.close()