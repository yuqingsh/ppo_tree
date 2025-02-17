import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from env import TreeHarvestEnv

# 初始化环境
env = TreeHarvestEnv()

check_env(env, warn=True, skip_render_check=True)
