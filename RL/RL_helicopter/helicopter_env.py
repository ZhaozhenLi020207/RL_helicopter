"""
直升机自动牵引入库环境 - 高速优化版
运动方向：从降落区域（y=3.55）向机库（y=0）运动
轨道：直线段1(机库侧 x=0) → 弯道(向右偏移到0.38) → 直线段2(降落区域侧 x=0.38)
【优化：移除所有渲染开销，只保留核心仿真逻辑】
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 禁用matplotlib渲染（完全移除）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免任何GUI开销

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Arc
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class HelicopterInboundKinematicsEnv(gym.Env):
    """
    直升机自动牵引入库环境 - 高速版
    运动方向：从降落区域（y=3.55）向机库（y=0）运动
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, fast_mode=True):
        super().__init__()

        # ========== 几何参数 ==========
        self.L_PFM = 0.198  # 牵引杆P到前轮中点FM的距离 (m)
        self.L_AFM = 1.2  # 尾轮A到前轮中点FM的距离 (m)
        self.L_PA = self.L_AFM - self.L_PFM

        # ========== 控制参数 ==========
        self.VX_MAX = 0.004  # 最大横移速度 (m/s)
        self.VY_MAX = 0.03  # 最大牵引速度 (m/s)
        self.VY_MIN = 0.005  # 最小牵引速度 (m/s)

        # ========== 轨道参数 ==========
        # 坐标系：y从0（机库）到3.55（降落区域）
        # 直升机从 y=3.55 向 y=0 运动
        self.y_start = 3.55  # 起点（降落区域）
        self.y_end = 0.0  # 终点（机库）
        self.y_curve_start = 1.4  # 弯道起点（从机库算起）
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
        self.max_steps = 5000
        self.trajectory = []
        self.fast_mode = fast_mode  # 快速模式：减少轨迹记录

        self.last_e_fm = None
        self.last_theta_rel = None
        self.last_y = None
        self.last_vx_cmd = None

        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        # 机库尺寸（能包住直升机）
        self.hangar_width = 0.8  # 机库宽度 (m)
        self.hangar_height = 1.5  # 机库高度 (m)

        # 预计算轨道中心线（加速）
        self._precompute_track()

    def _precompute_track(self):
        """预计算轨道中心线和导数（加速）"""
        self._track_cache = {}
        self._track_deriv_cache = {}
        self._track_angle_cache = {}

        # 预计算关键y值
        y_values = np.linspace(-0.5, self.y_start + 0.5, 1000)
        for y in y_values:
            self._track_cache[y] = self._track_centerline_calc(y)
            self._track_deriv_cache[y] = self._track_derivative_calc(y)
            self._track_angle_cache[y] = np.arctan2(self._track_deriv_cache[y], 1.0)

    def _track_centerline_calc(self, y):
        """轨道中心线函数（计算版本）"""
        if y <= self.y_curve_start:
            return 0.0
        elif y <= self.y_curve_end:
            t = (y - self.y_curve_start) / (self.y_curve_end - self.y_curve_start)
            return self.curve_dx * (1 - np.cos(np.pi * t)) / 2
        else:
            return self.curve_dx

    def _track_derivative_calc(self, y):
        """轨道中心线导数 dx/dy（计算版本）"""
        if y <= self.y_curve_start:
            return 0.0
        elif y <= self.y_curve_end:
            t = (y - self.y_curve_start) / (self.y_curve_end - self.y_curve_start)
            dt_dy = 1 / (self.y_curve_end - self.y_curve_start)
            dx_dt = self.curve_dx * np.pi * np.sin(np.pi * t) / 2
            return dx_dt * dt_dy
        else:
            return 0.0

    def track_centerline(self, y):
        """轨道中心线函数（带缓存）"""
        # 四舍五入到最近0.01米以使用缓存
        y_key = round(y * 100) / 100
        if y_key in self._track_cache:
            return self._track_cache[y_key]
        else:
            return self._track_centerline_calc(y)

    def track_derivative(self, y):
        """轨道中心线导数 dx/dy（带缓存）"""
        y_key = round(y * 100) / 100
        if y_key in self._track_deriv_cache:
            return self._track_deriv_cache[y_key]
        else:
            return self._track_derivative_calc(y)

    def track_angle(self, y):
        """轨道切线角度 (rad)（带缓存）"""
        y_key = round(y * 100) / 100
        if y_key in self._track_angle_cache:
            return self._track_angle_cache[y_key]
        else:
            return np.arctan2(self.track_derivative(y), 1.0)

    def update_kinematics(self, vx_cmd, vy_cmd, dt):
        """
        修正版运动学更新 - 确保vy_cmd正确转化为向机库速度
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 直接计算前轮中点速度
        # 横向速度分量
        v_fm_x = vx_cmd * cos_theta

        # 纵向速度分量（向机库，y减小）
        # 关键修正：vy_cmd 直接作为向前速度
        v_fm_y = -vy_cmd * cos_theta

        # 角速度：横向速度引起的旋转
        omega = vx_cmd / self.L_PFM
        omega = np.clip(omega, -0.5, 0.5)

        # 更新位置
        self.x_fm += v_fm_x * dt
        self.y_fm += v_fm_y * dt
        self.theta += omega * dt

        # 边界限制
        self.x_fm = np.clip(self.x_fm, -0.5, 1.0)
        self.y_fm = np.clip(self.y_fm, -0.5, self.y_start + 0.5)
        self.theta = np.clip(self.theta, -np.pi / 3, np.pi / 3)

        # 更新牵引杆位置
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 尾轮偏角
        self.tail_angle = np.clip(self.tail_angle + omega * dt * 0.5, -0.3, 0.3)

    def compute_reward(self, e_fm, theta_rel, y_fm):
        """
        模块化奖励函数 - 基于多种奖励策略设计

        策略清单：
        1. 势能函数（距离终点）
        2. 塑形奖励（偏差、偏角）
        3. 相对改善奖励
        4. 分阶段权重
        5. 弯道跟随奖励
        6. 终点高精度奖励
        """

        # ========== 可调参数 ==========
        # 权重系数
        W_PROGRESS = 2.0  # 前进奖励权重
        W_ALIVE = 0.1  # 存活奖励
        W_DEVIATION = 3.0  # 偏差惩罚权重
        W_ANGLE = 2.0  # 偏角惩罚权重
        W_IMPROVE_E = 1.5  # 偏差改善奖励
        W_IMPROVE_THETA = 2.0  # 偏角改善奖励
        W_CURVE_FOLLOW = 2.0  # 弯道跟随奖励
        W_TERMINAL = 10.0  # 终点接近奖励

        # 阶段阈值
        CURVE_START = 1.4
        CURVE_END = 2.65
        TERMINAL_START = 0.5

        # 成功阈值
        SUCCESS_E = 0.05
        SUCCESS_THETA = 0.05  # ~2.86°
        PERFECT_E = 0.03
        PERFECT_THETA = 0.02  # ~1.15°

        reward = 0.0

        # ========== 1. 势能函数（基于距离终点） ==========
        # 距离终点越近势能越高
        potential = -y_fm  # 负距离
        # 注：势能差在改善奖励中体现

        # ========== 2. 塑形奖励 - 偏差惩罚 ==========
        deviation_penalty = -W_DEVIATION * abs(e_fm)
        reward += deviation_penalty

        # ========== 3. 塑形奖励 - 偏角惩罚 ==========
        angle_penalty = -W_ANGLE * abs(theta_rel)
        reward += angle_penalty

        # ========== 4. 前进奖励（稠密） ==========
        # 基于y位置的前进奖励
        progress = (self.y_start - y_fm) / self.y_start
        reward += W_PROGRESS * progress

        # ========== 5. 相对改善奖励 ==========
        if self.last_e_fm is not None:
            # 偏差改善
            e_improve = abs(self.last_e_fm) - abs(e_fm)
            if e_improve > 0:
                reward += W_IMPROVE_E * e_improve
            elif e_improve < -0.02:
                reward -= 0.5

            # 偏角改善
            theta_improve = abs(self.last_theta_rel) - abs(theta_rel)
            if theta_improve > 0:
                reward += W_IMPROVE_THETA * theta_improve
            elif theta_improve < -0.02:
                reward -= 0.5

        # ========== 6. 存活奖励 ==========
        reward += W_ALIVE

        # ========== 7. 分阶段权重 - 弯道区域 ==========
        if CURVE_START < y_fm < CURVE_END:
            # 弯道区域：提高偏角重要性
            track_angle = self.track_angle(y_fm)
            angle_diff = abs(theta_rel - track_angle)

            # 弯道跟随奖励
            if angle_diff < 0.05:  # ~2.86°
                reward += W_CURVE_FOLLOW * 2.0
            elif angle_diff < 0.10:  # ~5.73°
                reward += W_CURVE_FOLLOW * 1.0
            elif angle_diff < 0.15:  # ~8.59°
                reward += W_CURVE_FOLLOW * 0.5

            # 弯道区域额外偏角惩罚
            reward -= 1.0 * abs(theta_rel)

        # ========== 8. 分阶段权重 - 终点区域 ==========
        if y_fm < TERMINAL_START:
            # 终点区域：提高偏差重要性，增加精度奖励
            terminal_factor = (TERMINAL_START - y_fm) / TERMINAL_START

            # 精度奖励（随距离指数增长）
            e_bonus = W_TERMINAL * (1 - min(1.0, abs(e_fm) / 0.1)) * terminal_factor
            theta_bonus = W_TERMINAL * 0.8 * (1 - min(1.0, abs(theta_rel) / 0.17)) * terminal_factor
            reward += e_bonus + theta_bonus

            # 终点区域额外偏差惩罚
            reward -= 2.0 * abs(e_fm) * (1 + terminal_factor)

            # 鼓励减速（可选）
            if hasattr(self, 'last_vx_cmd') and self.last_vx_cmd is not None:
                # 速度越小奖励越大
                speed_penalty = -0.5 * abs(self.last_vx_cmd)
                reward += speed_penalty

        # ========== 9. 防止超限惩罚（软约束） ==========
        # 偏差接近0.5m时开始惩罚
        if abs(e_fm) > 0.4:
            overflow = (abs(e_fm) - 0.4) / 0.1  # 0-1之间
            reward -= 5.0 * overflow

        # 偏角接近45°时开始惩罚
        if abs(theta_rel) > np.deg2rad(35):
            overflow = (abs(theta_rel) - np.deg2rad(35)) / np.deg2rad(10)
            reward -= 8.0 * overflow

        # ========== 10. 成功/终止奖励 ==========
        if y_fm <= self.y_end:
            # 到达终点
            if abs(e_fm) < SUCCESS_E and abs(theta_rel) < SUCCESS_THETA:
                # 成功入库
                if abs(e_fm) < PERFECT_E and abs(theta_rel) < PERFECT_THETA:
                    reward += 1000  # 完美入库
                    if not self.fast_mode:
                        print(f"🏆 完美入库！e={e_fm:.3f}m, θ={np.rad2deg(theta_rel):.1f}°")
                else:
                    reward += 500  # 成功入库
                    if not self.fast_mode:
                        print(f"✓ 成功入库！e={e_fm:.3f}m, θ={np.rad2deg(theta_rel):.1f}°")
            else:
                # 到达但精度不够
                reward -= 100
                if not self.fast_mode:
                    print(f"✗ 到达但精度不足 | e={e_fm:.3f}m, θ={np.rad2deg(theta_rel):.1f}°")

        # ========== 11. 记录状态供下一步使用 ==========
        self.last_e_fm = e_fm
        self.last_theta_rel = theta_rel
        self.last_y = y_fm

        return reward

    def reset(self, seed=None, options=None):
        """重置环境 - 只有正常模式"""
        super().reset(seed=seed)

        # 正常模式：从起点附近开始（y接近3.55）
        self.y_fm = self.np_random.uniform(2.8, self.y_start)
        e_fm_init = self.np_random.uniform(-0.25, 0.25)
        self.theta = self.np_random.uniform(-np.deg2rad(12), np.deg2rad(12))

        track_x = self.track_centerline(self.y_fm)
        self.x_fm = track_x + e_fm_init

        self.tail_angle = 0.0
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        self.current_steps = 0

        # 快速模式下减少轨迹记录
        if not self.fast_mode:
            self.trajectory = []
        else:
            self.trajectory = []  # 仍然初始化，但记录频率降低

        obs = self._get_obs()
        self.last_e_fm = obs[0]
        self.last_theta_rel = obs[1]
        self.last_y = self.y_fm
        self.last_vx_cmd = 0.0

        if not self.fast_mode:
            self._record_state()

        return obs, {}

    def _get_obs(self):
        """获取观测值"""
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        ideal_x_p = self.track_centerline(self.y_fm) - self.L_PFM * np.sin(self.track_angle(self.y_fm))
        e_p = self.x_p - ideal_x_p

        y_remaining = self.y_fm - self.y_end  # 剩余距离到机库

        return np.array([
            e_fm,
            theta_rel,
            e_p,
            self.tail_angle,
            y_remaining
        ], dtype=np.float32)

    def _record_state(self):
        """记录状态 - 快速模式下减少记录频率"""
        if self.fast_mode:
            # 每10步记录一次，减少内存和计算
            if self.current_steps % 10 == 0:
                self.trajectory.append({
                    'x_fm': self.x_fm,
                    'y_fm': self.y_fm,
                    'theta': self.theta,
                    'x_p': self.x_p,
                    'y_p': self.y_p,
                    'tail_angle': self.tail_angle
                })
        else:
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
        self.last_vx_cmd = vx_cmd

        self.update_kinematics(vx_cmd, vy, dt)
        self.current_steps += 1

        if not self.fast_mode:
            self._record_state()

        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        reward = self.compute_reward(e_fm, theta_rel, self.y_fm)

        self.last_e_fm = e_fm
        self.last_theta_rel = theta_rel

        # 终止条件：到达机库 (y_fm <= 0)
        terminated = self.y_fm <= self.y_end

        # 失败条件
        failed = (abs(e_fm) > 0.5 and abs(theta_rel) > np.deg2rad(45)) or \
                 self.current_steps >= self.max_steps

        truncated = failed

        if failed and not terminated:
            reward -= 15

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """渲染 - 快速模式下禁用"""
        if self.fast_mode:
            return

        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 14))

        self.ax.clear()

        # 设置坐标轴
        self.ax.set_xlim(-0.5, 1.0)
        self.ax.set_ylim(-0.5, self.y_start + 0.5)
        self.ax.set_xlabel("横向位置 x (m)", fontsize=12)
        self.ax.set_ylabel("纵向位置 y (m)", fontsize=12)
        self.ax.set_title("直升机自动牵引入库仿真", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # 绘制轨道中心线
        ys = np.linspace(-0.5, self.y_start + 0.5, 500)
        xs = [self.track_centerline(y) for y in ys]
        self.ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道', zorder=2)

        # 绘制轨道边界
        self.ax.fill_between(ys,
                             [x - 0.1 for x in xs],
                             [x + 0.1 for x in xs],
                             color='blue', alpha=0.15, zorder=0, label='轨道边界')

        # 绘制弯道区域
        self.ax.axhspan(self.y_curve_start, self.y_curve_end,
                        alpha=0.3, color='yellow', zorder=0,
                        label=f'弯道区域 ({self.y_curve_start}-{self.y_curve_end}m)')

        # ========== 机库（能包住直升机） ==========
        hangar_x_center = self.track_centerline(self.y_end)
        hangar_rect = Rectangle((hangar_x_center - self.hangar_width / 2, -1.2),
                                self.hangar_width, self.hangar_height,
                                linewidth=2, edgecolor='red',
                                facecolor='lightcoral', alpha=0.5, zorder=1)
        self.ax.add_patch(hangar_rect)
        self.ax.text(hangar_x_center, -0.5, '机库', fontsize=12,
                     ha='center', va='center', color='red', fontweight='bold')

        # 机库门框
        door_arc = Arc((hangar_x_center, -0.3), 0.5, 0.5, angle=0, theta1=0, theta2=180,
                       linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(door_arc)

        # 起点标注（降落区域）
        start_x = self.track_centerline(self.y_start)
        self.ax.plot(start_x, self.y_start, 'gs', markersize=12, label='起点', zorder=3)
        self.ax.text(start_x + 0.08, self.y_start, '降落区域', fontsize=11,
                     color='green', fontweight='bold')

        # 方向箭头（从起点指向机库，向下）
        self.ax.annotate('', xy=(0.2, 0.5), xytext=(0.2, self.y_start - 0.5),
                         arrowprops=dict(arrowstyle='->', color='black', lw=2))
        self.ax.text(0.25, self.y_start / 2, '牵引方向\n(向机库)', fontsize=10,
                     rotation=90, va='center', ha='center')

        # 绘制直升机
        self._draw_helicopter()

        # 绘制轨迹
        if len(self.trajectory) > 1:
            fm_traj_x = [t['x_fm'] for t in self.trajectory[::3]]
            fm_traj_y = [t['y_fm'] for t in self.trajectory[::3]]
            self.ax.plot(fm_traj_x, fm_traj_y, 'g--', linewidth=1.5,
                         alpha=0.6, label='前轮轨迹', zorder=1)

        # 显示信息
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        theta_rel = self.theta - self.track_angle(self.y_fm)

        info_text = f"步数: {self.current_steps}\n"
        info_text += f"位置: {self.y_fm:.2f}/{self.y_start:.2f}m\n"
        info_text += f"进度: {(self.y_start - self.y_fm) / self.y_start * 100:.1f}%\n"
        info_text += f"偏差: {e_fm:.3f}m\n"
        info_text += f"偏角: {np.rad2deg(theta_rel):.1f}°\n"
        info_text += f"尾轮角: {np.rad2deg(self.tail_angle):.1f}°"

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white',
                               edgecolor='gray', alpha=0.85))

        self.ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def _draw_helicopter(self):
        """绘制直升机"""
        fm_x, fm_y = self.x_fm, self.y_fm

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # 机头（前轮中点）
        nose = (fm_x, fm_y)

        # 机尾
        tail_x = fm_x - self.L_AFM * sin_theta
        tail_y = fm_y - self.L_AFM * cos_theta

        # 机身三角形
        body_width = 0.12
        left_rear = (tail_x - body_width * cos_theta, tail_y + body_width * sin_theta)
        right_rear = (tail_x + body_width * cos_theta, tail_y - body_width * sin_theta)

        body = Polygon([nose, left_rear, right_rear],
                       closed=True, color='gray', alpha=0.85, zorder=2)
        self.ax.add_patch(body)

        # 前轮
        front_wheel = Circle(nose, 0.05, color='black', zorder=3)
        self.ax.add_patch(front_wheel)

        # 尾轮
        tail_wheel = Circle((tail_x, tail_y), 0.045, color='red', zorder=3)
        self.ax.add_patch(tail_wheel)

        # 尾轮朝向指示线
        tail_wheel_angle = self.theta + self.tail_angle
        arrow_len = 0.12
        arrow_end_x = tail_x + arrow_len * np.sin(tail_wheel_angle)
        arrow_end_y = tail_y + arrow_len * np.cos(tail_wheel_angle)
        self.ax.annotate('', xy=(arrow_end_x, arrow_end_y),
                         xytext=(tail_x, tail_y),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # 牵引杆
        towbar_x = self.x_p
        towbar_y = self.y_p
        self.ax.plot([towbar_x, fm_x], [towbar_y, fm_y],
                     linewidth=2.5, color='orange', zorder=2)
        self.ax.plot(towbar_x, towbar_y, 'ro', markersize=5, zorder=3)

        # 理想牵引杆位置
        ideal_x = self.track_centerline(self.y_fm) - self.L_PFM * np.sin(self.track_angle(self.y_fm))
        self.ax.plot(ideal_x, self.y_fm, 'g*', markersize=8, alpha=0.7, zorder=2)

    def make_animation(self, filename="helicopter_inbound.gif", save_frames=200):
        """生成动画"""
        if len(self.trajectory) == 0:
            print("无轨迹数据")
            return

        fig, ax = plt.subplots(figsize=(12, 14))

        def animate(frame_idx):
            ax.clear()

            t = self.trajectory[min(frame_idx, len(self.trajectory) - 1)]
            x_fm, y_fm = t['x_fm'], t['y_fm']
            theta = t['theta']
            x_p, y_p = t['x_p'], t['y_p']

            ax.set_xlim(-0.5, 1.0)
            ax.set_ylim(-0.5, self.y_start + 0.5)
            ax.set_xlabel("横向位置 x (m)")
            ax.set_ylabel("纵向位置 y (m)")
            ax.set_title(f"直升机自动牵引入库 (步数: {frame_idx})")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # 绘制轨道
            ys = np.linspace(-0.5, self.y_start + 0.5, 500)
            xs = [self.track_centerline(y) for y in ys]
            ax.plot(xs, ys, 'b-', linewidth=3, label='牵引轨道')
            ax.fill_between(ys, [x - 0.1 for x in xs], [x + 0.1 for x in xs],
                            color='blue', alpha=0.15)

            # 弯道区域
            ax.axhspan(self.y_curve_start, self.y_curve_end,
                       alpha=0.2, color='yellow')

            # 机库
            hangar_x_center = self.track_centerline(self.y_end)
            hangar_rect = Rectangle((hangar_x_center - self.hangar_width / 2, -1.2),
                                    self.hangar_width, self.hangar_height,
                                    linewidth=2, edgecolor='red',
                                    facecolor='lightcoral', alpha=0.5)
            ax.add_patch(hangar_rect)
            ax.text(hangar_x_center, -0.5, '机库', fontsize=12, ha='center', color='red')

            # 起点
            start_x = self.track_centerline(self.y_start)
            ax.plot(start_x, self.y_start, 'gs', markersize=10)
            ax.text(start_x + 0.08, self.y_start, '起点', fontsize=10, color='green')

            # 历史轨迹
            traj_x = [t_hist['x_fm'] for t_hist in self.trajectory[:frame_idx + 1:3]]
            traj_y = [t_hist['y_fm'] for t_hist in self.trajectory[:frame_idx + 1:3]]
            ax.plot(traj_x, traj_y, 'g--', linewidth=1.5, alpha=0.6)

            # 绘制直升机
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            nose = (x_fm, y_fm)
            tail_x = x_fm - self.L_AFM * sin_theta
            tail_y = y_fm - self.L_AFM * cos_theta
            left_rear = (tail_x - 0.12 * cos_theta, tail_y + 0.12 * sin_theta)
            right_rear = (tail_x + 0.12 * cos_theta, tail_y - 0.12 * sin_theta)

            body = Polygon([nose, left_rear, right_rear],
                           closed=True, color='gray', alpha=0.85)
            ax.add_patch(body)

            ax.plot([x_p, x_fm], [y_p, y_fm], 'orange', linewidth=2.5)
            ax.plot(x_p, y_p, 'ro', markersize=5)

            e_fm = x_fm - self.track_centerline(y_fm)
            info_text = f"偏差: {e_fm:.3f}m | y: {y_fm:.2f}m"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax.legend(loc='upper left')
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

