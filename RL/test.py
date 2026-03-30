# BUAA  ZY2407706
# Author：LZZ
# Date： 2026/3/10  22:48
import gymnasium as gym
import logging

logging.basicConfig(
    level=logging.INFO,  # 关键：将日志级别设为INFO（显示INFO及以上级别日志）
    format='%(levelname)s: %(message)s'  # 简化输出格式，也可保留默认
)

env = gym.make('MountainCar-v0')
logging.info("=== env.spec 属性 ===")
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
logging.info("\n=== env.unwrapped 属性 ===")
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

env.close()