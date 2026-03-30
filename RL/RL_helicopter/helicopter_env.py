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

        self.L_PFM = 3.0  # 牵引杆P到前轮中点FM的距离
        self.max_steps = 500  # 最大入库步数
        self.render_mode = render_mode

        # 轨道参数
        self.y_start = 30.0  # 初始位置（直道1起点）
        self.y_curve_start = 20.0  # 弯道起点
        self.y_curve_end = 10.0  # 弯道终点（直道2起点）
        self.y_hangar_end = 0.0  # 机库终点
        self.curve_dx = 5.0  # 弯道最大横移偏移

        # 动作空间
        # 动作定义：0=左横移(-0.05m/s)+慢牵(0.1m/s)，1=无横移+中牵(0.3m/s)，2=右横移(+0.05m/s)+快牵(0.5m/s)
        self.action_space = spaces.Discrete(3)
        self.vx_options = [-0.05, 0.0, 0.05]  # 横移速度（垂直轨道）
        self.vy_options = [0.1, 0.3, 0.5]    # 牵引速度（沿轨道，向机库移动y减小）

        # 观测空间
        # 观测维度：[前轮相对轨道偏差, 机身偏角, 牵引杆相对轨道偏差]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -np.pi/15, -1.0], dtype=np.float32),  # 偏差±1m，偏角±12°
            high=np.array([1.0, np.pi/15, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 状态变量
        self.x_fm = None  # 前轮中点FM的x坐标（横移）
        self.y_fm = None  # 前轮中点FM的y坐标（沿轨道）
        self.theta = None  # 机身偏角（论文定义：逆时针为正）
        self.x_p = None   # 牵引杆P的x坐标（横移）
        self.y_p = None   # 牵引杆P的y坐标（沿轨道）
        self.success_frames = []  # 成功入库帧记录

    # 轨道中心线函数
    def track_centerline(self, y):
        """计算y位置对应的轨道中心线x坐标"""
        if y >= self.y_curve_start:
            # 直道1：x=0
            return 0.0
        elif y >= self.y_curve_end:
            # 弯道：余弦平滑过渡（论文5.5.2优化轨道）
            t = (self.y_curve_start - y) / (self.y_curve_start - self.y_curve_end)
            return self.curve_dx * (1 - np.cos(np.pi * t)) / 2
        else:
            # 直道2（机库段）：x=curve_dx
            return self.curve_dx

    # 重置环境（初始位姿）
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps = 0
        self.success_frames = []

        # 初始位姿：直道1上，前轮中点在轨道附近偏移
        self.y_fm = self.np_random.uniform(self.y_start - 2.0, self.y_start)
        track_x = self.track_centerline(self.y_fm)
        self.x_fm = track_x + self.np_random.uniform(-0.6, 0.6)  # 初始横移偏差±0.6m
        self.theta = self.np_random.uniform(-np.pi/15, np.pi/15)  # 初始偏角±12°

        # 牵引杆P的初始位置
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 观测值：相对轨道的偏差
        obs = np.array([
            self.x_fm - track_x,  # 前轮相对轨道偏差e_fm
            self.theta,           # 机身偏角θ
            self.x_p - self.track_centerline(self.y_p)  # 牵引杆相对轨道偏差e_p
        ], dtype=np.float32)
        return obs, {}

    # 步进函数
    def step(self, action):
        self.current_steps += 1
        # 1. 解析动作（横移速度vx，牵引速度vy）
        vx = self.vx_options[action]
        vy = self.vy_options[action]

        # 2. 运动学更新
        dt = 0.1  # 控制步长
        # 机身角速度（耦合vx、vy、theta）
        omega = (vx * np.cos(self.theta) - vy * np.sin(self.theta)) / self.L_PFM
        # 更新前轮位置
        self.x_fm += (vx - self.L_PFM * omega * np.sin(self.theta)) * dt
        self.y_fm -= vy * dt  # 向机库移动，y减小
        # 更新机身偏角（限制在±12°）
        self.theta += omega * dt
        self.theta = np.clip(self.theta, -np.pi/15, np.pi/15)
        # 更新牵引杆位置（几何关系）
        self.x_p = self.x_fm + self.L_PFM * np.sin(self.theta)
        self.y_p = self.y_fm - self.L_PFM * np.cos(self.theta)

        # 3. 计算相对轨道偏差（控制目标）
        e_fm = self.x_fm - self.track_centerline(self.y_fm)  # 前轮偏差
        e_p = self.x_p - self.track_centerline(self.y_p)    # 牵引杆偏差

        # 4. 奖励函数（引导平稳入库）
        reward = 0.0
        # 惩罚偏差：前轮偏差权重1，牵引杆偏差权重0.5，机身偏角权重2（论文优先级）
        reward -= abs(e_fm) + 0.5 * abs(e_p) + 2 * abs(self.theta)
        # 奖励速度：牵引速度越快，奖励越高（鼓励快速入库）
        reward += 0.1 * vy
        # 入库成功大奖（指标：e_fm<0.05m，theta<1°，且进入机库y_fm<5m）
        if abs(e_fm) < 0.05 and abs(self.theta) < 0.017 and self.y_fm < 5.0:
            reward += 50.0
        # 失稳惩罚（偏角超12°或偏差超1m）
        if abs(self.theta) > np.pi/15 or abs(e_fm) > 1.0:
            reward -= 10.0

        # 5. 终止条件
        terminated = (abs(e_fm) < 0.05 and abs(self.theta) < 0.017 and self.y_fm < 5.0)  # 入库成功
        truncated = (self.current_steps >= self.max_steps) or (abs(e_fm) > 1.0)  # 超时或失稳

        # 6. 记录成功帧
        if terminated:
            self.success_frames.append((self.x_fm, self.y_fm, self.theta, self.x_p, self.y_p))

        # 7. 输出观测值
        obs = np.array([e_fm, self.theta, e_p], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    # 可视化
    def make_animation(self, path="helicopter_inbound_success.gif"):
        if not self.success_frames:
            print("无成功入库数据，无法生成动画！")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-1.5, 2.5)  # 横移范围
        ax.set_ylim(-2.0, 35.0)  # 沿轨道范围
        ax.set_title("直升机牵引入库成功动画（论文运动学模型）")
        ax.set_xlabel("横移位置 (m)")
        ax.set_ylabel("沿轨道位置 (m)")
        ax.grid(True)

        # 1. 绘制轨道（直-弯-直）
        ys = np.linspace(self.y_hangar_end, self.y_start, 300)
        xs = [self.track_centerline(y) for y in ys]
        ax.plot(xs, ys, 'k-', lw=2, label="牵引轨道")

        # 2. 绘制机库（定义：直道2末端，x∈[0.0,1.0], y∈[0.0,5.0]）
        hangar = Rectangle((0.0, 0.0), 1.0, 5.0, linewidth=2, edgecolor='red', facecolor='none', label="机库")
        ax.add_patch(hangar)

        # 3. 绘制直升机（黑色三角形+三个红点轮胎，你的要求）
        tri = Polygon(np.zeros((3, 2)), closed=True, color='black', alpha=0.8, label="直升机机身")
        ax.add_patch(tri)
        # 轮胎：前轮到机身前端1m，左右轮间距0.8m（机身尺寸）
        t_front = Circle((0, 0), 0.12, color='red', label="前轮")
        t_left = Circle((0, 0), 0.12, color='red', label="左后轮")
        t_right = Circle((0, 0), 0.12, color='red', label="右后轮")
        ax.add_patch(t_front)
        ax.add_patch(t_left)
        ax.add_patch(t_right)

        # 4. 绘制牵引杆（连接牵引杆P与机身）
        towbar, = ax.plot([], [], 'k-', lw=2, label="牵引杆")

        # 5. 绘制运动轨迹（前轮轨迹）
        trajectory_line, = ax.plot([], [], 'g--', lw=1, label="前轮轨迹")

        ax.legend(loc="upper right")

        def update(frame):
            x_fm, y_fm, theta, x_p, y_p = frame
            # 机身三个顶点：前端（前轮前方1m）、左后端、右后端
            front_x = x_fm + 1.0 * np.cos(theta)
            front_y = y_fm + 1.0 * np.sin(theta)
            left_rear_x = x_fm - 2.0 * np.cos(theta) - 0.4 * np.sin(theta)
            left_rear_y = y_fm - 2.0 * np.sin(theta) + 0.4 * np.cos(theta)
            right_rear_x = x_fm - 2.0 * np.cos(theta) + 0.4 * np.sin(theta)
            right_rear_y = y_fm - 2.0 * np.sin(theta) - 0.4 * np.cos(theta)

            # 更新机身和轮胎位置
            tri.set_xy([(front_x, front_y), (left_rear_x, left_rear_y), (right_rear_x, right_rear_y)])
            t_front.set_center((x_fm, y_fm))
            t_left.set_center((left_rear_x, left_rear_y))
            t_right.set_center((right_rear_x, right_rear_y))

            # 更新牵引杆（连接牵引杆P与机身中心）
            body_center_x = (front_x + left_rear_x + right_rear_x) / 3
            body_center_y = (front_y + left_rear_y + right_rear_y) / 3
            towbar.set_data([x_p, body_center_x], [y_p, body_center_y])

            # 更新轨迹
            traj_xs = [f[0] for f in self.success_frames[:self.success_frames.index(frame)+1]]
            traj_ys = [f[1] for f in self.success_frames[:self.success_frames.index(frame)+1]]
            trajectory_line.set_data(traj_xs, traj_ys)

            return tri, t_front, t_left, t_right, towbar, trajectory_line

        ani = FuncAnimation(fig, update, frames=self.success_frames, interval=80, blit=True)
        ani.save(path, writer='pillow', fps=10)
        plt.close(fig)
        print(f"✅ 入库动画已保存至：{path}")

    def render(self):
        """实时渲染（训练时可选开启）"""
        if self.render_mode != "human":
            return
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-2.0, 35.0)
        ax.plot([self.track_centerline(y) for y in np.linspace(0, 30, 200)], np.linspace(0, 30, 200), 'k-', lw=2)
        ax.add_patch(Rectangle((0.0, 0.0), 1.0, 5.0, edgecolor='red', facecolor='none'))
        # 绘制直升机
        front_x = self.x_fm + 1.0 * np.cos(self.theta)
        front_y = self.y_fm + 1.0 * np.sin(self.theta)
        left_rear_x = self.x_fm - 2.0 * np.cos(self.theta) - 0.4 * np.sin(self.theta)
        left_rear_y = self.y_fm - 2.0 * np.sin(self.theta) + 0.4 * np.cos(self.theta)
        right_rear_x = self.x_fm - 2.0 * np.cos(self.theta) + 0.4 * np.sin(self.theta)
        right_rear_y = self.y_fm - 2.0 * np.sin(self.theta) - 0.4 * np.cos(self.theta)
        ax.fill([front_x, left_rear_x, right_rear_x], [front_y, left_rear_y, right_rear_y], 'black', alpha=0.8)
        ax.scatter([self.x_fm, left_rear_x, right_rear_x], [self.y_fm, left_rear_y, right_rear_y], c='red', s=30)
        # 绘制牵引杆
        body_center_x = (front_x + left_rear_x + right_rear_x) / 3
        body_center_y = (front_y + left_rear_y + right_rear_y) / 3
        ax.plot([self.x_p, body_center_x], [self.y_p, body_center_y], 'k-', lw=2)
        plt.xlabel("横移位置 (m)")
        plt.ylabel("沿轨道位置 (m)")
        plt.title(f"入库实时状态：步数={self.current_steps} | 前轮偏差={self.x_fm - self.track_centerline(self.y_fm):.3f}m | 偏角={np.degrees(self.theta):.1f}°")
        plt.grid(True)
        plt.pause(0.01)
        plt.clf()

    def close(self):
        plt.close('all')