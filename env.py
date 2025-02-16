import gymnasium as gym
from gymnasium import spaces
from utils import ForestManager
import numpy as np

MAX_TREE_CUT = 1500
GRID_SIZE = 20
CANOPY_CLOSURE_THRESHOLD = 0.7
W1 = 0.3
W2 = 0.3
W3 = 0.4


class TreeHarvestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.fm = ForestManager("classification.tif")
        self.stats = self.fm.get_stats()

        self.n_trees = len(self.fm.trees)
        self.left = self.fm.bounds.left
        self.bottom = self.fm.bounds.bottom
        self.width = self.fm.bounds.right - self.fm.bounds.left
        self.height = self.fm.bounds.top - self.fm.bounds.bottom

        single_tree_space = gym.spaces.Dict(
            {
                "max_chm": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "xiongjing": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "guanfu": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "x": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "is_cut": spaces.Discrete(2),
                "angle_index": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )
        self.observation_space = gym.spaces.Tuple(
            [single_tree_space for _ in range(self.n_trees)]
        )

        self.action_space = gym.spaces.MultiBinary(n=self.n_trees)

    def _get_single_observation(self, tree_id):
        tree = self.fm.trees.loc[tree_id]
        return {
            "max_chm": (tree["max_chm"] - self.stats["max_chm_min"])
            / (self.stats["max_chm_max"] - self.stats["max_chm_min"]),
            "xiongjing": (tree["xiongjing"] - self.stats["xiongjing_min"])
            / (self.stats["xiongjing_max"] - self.stats["xiongjing_min"]),
            "guanfu": (tree["guanfu"] - self.stats["guanfu_min"])
            / (self.stats["guanfu_max"] - self.stats["guanfu_min"]),
            "x": (tree["geometry"].x - self.left) / self.width,
            "y": (tree["geometry"].y - self.bottom) / self.height,
            "is_cut": tree["is_cut"],
            "angle_index": tree["angle_index"],
        }

    def _get_observation(self):
        return tuple([self._get_single_observation(i) for i in range(self.n_trees)])

    def _calculate_reward(self, action):

        tree = self._get_single_observation(action)
        health_score = tree["max_chm"] * tree["xiongjing"] * tree["guanfu"]

        angle_index_old = self.fm.get_sum_angle_index()
        self.fm.harvest_tree(action)
        angle_index_new = self.fm.get_sum_angle_index()
        delta_angle = angle_index_new - angle_index_old

        canopy_closure_penalty = -10 if self.fm.yubidu < CANOPY_CLOSURE_THRESHOLD else 0

        return -W1 * health_score + W2 * delta_angle + W3 * canopy_closure_penalty

    def reset(self):
        self.fm = ForestManager("classification.tif")
        return self._get_observation()

    def step(self, action):
        reward = self._calculate_reward(action)
        print(reward)
        done = (
            self.fm.trees["is_cut"].sum() >= MAX_TREE_CUT
            or self.fm.yubidu < CANOPY_CLOSURE_THRESHOLD
        )
        return self._get_observation(), reward, done, {}


if __name__ == "__main__":
    env = TreeHarvestEnv()
    env.reset()
    env.step(5)
