import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.animation import FuncAnimation

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class HelicopterInboundEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.L_PFM = 0.198  # 牵引杆P到前轮中点FM的距离
        self.max_steps = 5000  # 减小步数，避免超时
        self.render_mode = render_mode

        # 轨道参数（缩小场景，和可视化匹配）
        self.y_start = 3.0  # 初始位置（从3m开始，不是30m！）
        self.y_curve_start = 2.0  # 弯道起点
        self.y_curve_end = 1.0  # 弯道终点
        self.y_hangar_end = 0.0  # 机库终点
        self.curve_dx = 0.5  # 弯道横移（缩小到可视化范围）

        # 动作空间（平滑化，减小速度，避免激进）
        self.action_space = spaces.Discrete(3)
        self.vx_options = [-0.015, 0.0, 0.015]  # 横移速度减半，更平稳
        self.vy_options = [0.03, 0.06, 0.09]    # 牵引速度减半，大惯性不超速

        # 观测空间（修复阈值！和初始状态匹配）
        self.observation_space = spaces.Box(
            low=np.array([-0.1, -np.pi/15, -0.5], dtype=np.float32),  # 前轮偏差±0.1m
            high=np.array([0.1, np.pi/15, 0.5], dtype=np.float32),
            dtype=np.float32
        )

        # 状态变量
        self.x_fm = None  # 前轮中点FM的x坐标
        self.y_fm = None  # 前轮中点FM的y坐标
        self.theta = None  # 机身偏角
        self.x_p = None   # 牵引杆P的x坐标
        self.y_p = None   # 牵引杆P的y坐标
        self.current_steps = 0
        self.success_frames = []

    # 轨道中心线函数
    def track_centerline(self, y):
        if y >= self.y_curve_start:
            return 0.0
        elif y >= self.y_curve_end:
            t = (self.y_curve_start - y) / (self.y_curve_start - self.y_curve_end)
            return self.curve_dx * (1 - np.cos(np.pi * t)) / 2
        else:
            return self.curve_dx

    # 重置环境（缩小初始偏差，避免初始失稳）
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps = 0
        self.success_frames = []

        # 初始位姿：偏差缩小到±0.02m，偏角±6°（更安全）
        self.y_fm = self.np_random.uniform(self.y_start - 0.5, self.y_start)
        track_x = self.track_centerline(self.y_fm)
        self.x_fm = track_x + self.np_random.uniform(-0.02, 0.02)
        self.theta = self.np_random.uniform(-np.pi/30, np.pi/30)  # ±6°

        # 牵引杆P初始位置（正确几何关系）
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 观测值
        obs = np.array([
            self.x_fm - track_x,
            self.theta,
            self.x_p - self.track_centerline(self.y_p)
        ], dtype=np.float32)
        return obs, {}

    # 步进函数
    def step(self, action):
        self.current_steps += 1
        vx = self.vx_options[action]
        vy = self.vy_options[action]
        dt = 0.1

        # 运动学
        omega = (vx * np.cos(self.theta) + vy * np.sin(self.theta)) / self.L_PFM
        self.x_fm += (vx * np.cos(self.theta) - vy * np.sin(self.theta)) * dt
        self.y_fm -= vy * np.cos(self.theta) * dt
        self.theta += omega * dt
        self.theta = np.clip(self.theta, -np.pi / 15, np.pi / 15)
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 计算偏差
        e_fm = self.x_fm - self.track_centerline(self.y_fm)
        e_p = self.x_p - self.track_centerline(self.y_p)

        # 奖励函数
        # 1. 轻惩罚（别罚太狠）
        penalty = 2 * abs(e_fm) + 1 * abs(e_p) + 5 * abs(self.theta)
        # 2. 渐进奖励：向机库移动(y减小)就给小奖励，引导前进
        progress_reward = 0.5 if self.y_fm < self.y_curve_start else 0.1
        # 3. 总奖励：轻惩罚 + 前进奖励
        reward = -penalty + progress_reward

        # 放宽成功条件
        # 成功：偏差<0.1m + 偏角<2° + 进入机库区域(y<1.0)
        success_cond = (abs(e_fm) < 0.1 and abs(self.theta) < 0.035 and self.y_fm < 1.0)
        # 失稳：收紧，只在真失控时终止
        unstable_cond = (abs(self.theta) > np.pi / 15 or abs(e_fm) > 0.4)
        # =================================================================

        # 终止条件
        terminated = success_cond
        truncated = (self.current_steps >= self.max_steps) or unstable_cond

        # 成功大奖（足够大，正向激励）
        if terminated:
            reward += 200.0
            self.success_frames.append((self.x_fm, self.y_fm, self.theta, self.x_p, self.y_p))

        obs = np.array([e_fm, self.theta, e_p], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    # 可视化（完全匹配轨道场景，无坐标溢出）
    def make_animation(self, path="helicopter_inbound_success.gif"):
        if not self.success_frames:
            print("无成功入库数据，无法生成动画！")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-0.2, 1.0)  # 匹配轨道curve_dx=0.5
        ax.set_ylim(-0.2, 3.5)  # 匹配轨道y_start=3.0
        ax.set_title("直升机牵引入库成功图")
        ax.set_xlabel("横移位置 (m)")
        ax.set_ylabel("沿轨道位置 (m)")
        ax.grid(True)

        # 绘制轨道
        ys = np.linspace(self.y_hangar_end, self.y_start, 300)
        xs = [self.track_centerline(y) for y in ys]
        ax.plot(xs, ys, 'k-', lw=2, label="牵引轨道")

        # 机库
        hangar = Rectangle((0.3, 0.0), 0.4, 0.8, linewidth=2, edgecolor='red', facecolor='none', label="机库")
        ax.add_patch(hangar)

        # 直升机（倒三角，尖头朝后，尾轮）
        tri = Polygon(np.array([[0,0], [-0.09, -0.5], [0.09, -0.5]]), closed=True, color='black', alpha=0.8, label="机身")
        t_left = Circle((0,0), 0.06, color='red', label="尾轮")
        t_right = Circle((0,0), 0.06, color='red')
        t_back = Circle((0,0), 0.06, color='red')
        ax.add_patch(tri)
        ax.add_patch(t_left)
        ax.add_patch(t_right)
        ax.add_patch(t_back)

        # 牵引杆+轨迹
        towbar, = ax.plot([], [], 'k-', lw=2, label="牵引杆")
        trajectory_line, = ax.plot([], [], 'g--', lw=1, label="前轮轨迹")
        ax.legend(loc="upper right")

        def update(frame):
            x_fm, y_fm, theta, x_p, y_p = frame
            # 正确机身顶点
            front_x = x_fm
            front_y = y_fm
            left_x = x_fm - 0.09*np.cos(theta)
            left_y = y_fm - 0.5*np.sin(theta)
            right_x = x_fm + 0.09*np.cos(theta)
            right_y = y_fm - 0.5*np.sin(theta)

            tri.set_xy([(front_x, front_y), (left_x, left_y), (right_x, right_y)])
            t_left.set_center((left_x, left_y))
            t_right.set_center((right_x, right_y))
            t_back.set_center((x_fm - 0.5*np.cos(theta), y_fm - 0.5*np.sin(theta)))
            towbar.set_data([x_p, x_fm], [y_p, y_fm])

            traj_xs = [f[0] for f in self.success_frames[:self.success_frames.index(frame)+1]]
            traj_ys = [f[1] for f in self.success_frames[:self.success_frames.index(frame)+1]]
            trajectory_line.set_data(traj_xs, traj_ys)
            return tri, t_left, t_right, t_back, towbar, trajectory_line

        ani = FuncAnimation(fig, update, frames=self.success_frames, interval=80, blit=True)
        ani.save(path, writer='pillow', fps=10)
        plt.close(fig)
        print(f"✅ 入库动画已保存至：{path}")

    def render(self):
        if self.render_mode != "human":
            return
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_xlim(-0.2, 1.0)
        ax.set_ylim(-0.2, 3.5)
        # 轨道
        ys = np.linspace(0, 3, 200)
        xs = [self.track_centerline(y) for y in ys]
        ax.plot(xs, ys, 'k-', lw=2)
        # 机库
        ax.add_patch(Rectangle((0.3, 0.0), 0.4, 0.8, edgecolor='red', facecolor='none'))
        # 直升机
        front_x = self.x_fm
        front_y = self.y_fm
        left_x = self.x_fm - 0.09*np.cos(self.theta)
        left_y = self.y_fm - 0.5*np.sin(self.theta)
        right_x = self.x_fm + 0.09*np.cos(self.theta)
        right_y = self.y_fm - 0.5*np.sin(self.theta)
        ax.fill([front_x, left_x, right_x], [front_y, left_y, right_y], 'black', alpha=0.8)
        ax.scatter([left_x, right_x, front_x-0.5*np.cos(self.theta)],
                   [left_y, right_y, front_y-0.5*np.sin(self.theta)], c='red', s=20)
        # 牵引杆
        ax.plot([self.x_p, self.x_fm], [self.y_p, self.y_fm], 'k-', lw=2)
        plt.title(f"步数={self.current_steps} | 偏差={self.x_fm-self.track_centerline(self.y_fm):.3f}m | 偏角={np.degrees(self.theta):.1f}°")
        plt.grid(True)
        plt.pause(0.01)
        plt.clf()

    def close(self):
        plt.close('all')