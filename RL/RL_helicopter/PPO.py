# 训练代码示例（PPO算法）
from stable_baselines3 import PPO
from helicopter_env import HelicopterInboundEnv

if __name__ == "__main__":
    # 1. 创建环境（训练时可关闭render_mode，测试时开启）
    env = HelicopterInboundEnv(render_mode=None)

    # 2. 初始化PPO算法（适配论文控制目标）
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,  # 折扣因子（鼓励长期入库）
        verbose=1
    )

    # 3. 训练（至少5万步）
    print("开始训练直升机入库模型...")
    model.learn(total_timesteps=500000)

    # 4. 测试+生成动画
    print("开始测试，生成入库动画...")
    test_env = HelicopterInboundEnv(render_mode="human")
    obs, _ = test_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        if terminated or truncated:
            if terminated:
                test_env.make_animation("helicopter_inbound_success.gif")
            print("入库成功！" if terminated else "任务终止（超时/失稳）")
            break
    test_env.close()