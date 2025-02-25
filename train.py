import os
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TreeHarvestEnv
import gymnasium as gym


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


# 设置随机种子以确保结果可重复
SEED = 42
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
np.random.seed(SEED)

# 定义训练参数
TRAIN_STEPS = 100000  # 总训练步数
BATCH_SIZE = 32  # 每次更新的批大小
LEARNING_RATE = 0.05  # 学习率
CHECKPOINT_FREQ = 10000  # 每隔多少步保存一次模型

# 创建保存模型和日志的文件夹
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化环境
env = TreeHarvestEnv()
# env = Monitor(env)  # 包装环境以记录指标
# env = DummyVecEnv([lambda: env])  # 将单个环境包装为向量环境
env = ActionMasker(env, mask_fn)  # 包装环境以支持动作屏蔽
env = Monitor(env)  # 包装环境以记录指标

model = MaskablePPO(
    MaskableActorCriticPolicy,
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cpu",
    learning_rate=LEARNING_RATE,
    n_steps=BATCH_SIZE,
    batch_size=BATCH_SIZE,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,
    ent_coef=0.01,
    seed=SEED,
)
model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

# 定义PPO配置
"""
model = PPO(
    "MlpPolicy",  # 使用多层感知机策略网络
    env,
    learning_rate=LEARNING_RATE,
    n_steps=BATCH_SIZE,
    batch_size=BATCH_SIZE,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,
    ent_coef=0.01,
    verbose=1,  # 启用详细日志输出
    seed=SEED,
    tensorboard_log=LOG_DIR,
    device="cpu",
)

# 设置检查点回调函数
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ, save_path=SAVE_DIR, name_prefix="ppo_model"
)

# 开始训练
model.learn(
    total_timesteps=TRAIN_STEPS, callback=[checkpoint_callback], progress_bar=True
)

# 保存最终模型
model.save(os.path.join(SAVE_DIR, "ppo_final_model"))

# 关闭环境
env.close()
"""
