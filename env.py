import gymnasium as gym
from gymnasium import spaces
from utils import ForestManager
import numpy as np
from stable_baselines3.common.env_checker import check_env

MAX_TREE_CUT = 1500
GRID_SIZE = 20
CANOPY_CLOSURE_THRESHOLD = 0.7
W1 = 0.3
W2 = 0.7


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

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_trees * 6,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(self.n_trees)

    def _get_single_observation(self, tree_id):
        tree = self.fm.trees.loc[tree_id]
        if tree["is_cut"]:
            return {
                "max_chm": 0,
                "xiongjing": 0,
                "guanfu": 0,
                "x": 0,
                "y": 0,
                "is_cut": 1,
            }
        else:
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
            }

    def _get_observation(self):
        obs = []
        for i in range(self.n_trees):
            tree = self._get_single_observation(i)
            obs.extend(tree.values())

            height = tree["max_chm"]
            diameter = tree["xiongjing"]
            crown_width = tree["guanfu"]
            x = tree["x"]
            y = tree["y"]
            is_cut = float(tree["is_cut"])

            assert 0 <= height <= 1, f"Height out of range: {height}"
            assert 0 <= diameter <= 1, f"Diameter out of range: {diameter}"
            assert 0 <= crown_width <= 1, f"Crown width out of range: {crown_width}"
            assert 0 <= x <= 1, f"x coordinate out of range: {x}"
            assert 0 <= y <= 1, f"y coordinate out of range: {y}"
            assert is_cut in {0, 1}, f"is_cut must be 0 or 1: {is_cut}"

        obs = np.array(obs, dtype=np.float32)
        return obs

    def _calculate_reward(self, action):
        if self.fm.trees.loc[action, "is_cut"]:
            return -10

        tree = self._get_single_observation(action)
        health_score = tree["max_chm"] * tree["xiongjing"] * tree["guanfu"]
        compete_index = self.fm.get_compete_index(action)
        self.fm.harvest_tree(action)

        # canopy_closure_penalty = -10 if self.fm.yubidu < CANOPY_CLOSURE_THRESHOLD else 0

        return -W1 * health_score + W2 * compete_index  # + W3 * canopy_closure_penalty

    def reset(self, seed=0):
        np.random.seed(seed)
        self.fm = ForestManager("classification.tif")

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        reward = self._calculate_reward(action)
        done = self.fm.trees["is_cut"].sum() >= MAX_TREE_CUT
        obs = self._get_observation()
        return obs, reward, bool(done), False, {}


if __name__ == "__main__":
    env = TreeHarvestEnv()
    check_env(env, warn=True, skip_render_check=True)
