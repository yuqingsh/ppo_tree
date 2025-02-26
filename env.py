import gymnasium as gym
from gymnasium import spaces
from utils import ForestManager
import numpy as np
from stable_baselines3.common.env_checker import check_env

MAX_TREE_CUT = 50
CANOPY_CLOSURE_THRESHOLD = 0.7
W1 = 0.2
W2 = 0.2
W3 = 0.2
W4 = 0.4


class TreeHarvestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.fm = ForestManager()
        self.stats = self.fm.get_stats()

        self.n_trees = len(self.fm.trees)
        self.left = self.fm.bounds.left
        self.bottom = self.fm.bounds.bottom
        self.width = self.fm.bounds.right - self.fm.bounds.left
        self.height = self.fm.bounds.top - self.fm.bounds.bottom

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_trees * 5,), dtype=np.float32
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
            }
        else:
            obs = {
                "max_chm": (tree["max_chm"] - self.stats["max_chm_min"])
                / (self.stats["max_chm_max"] - self.stats["max_chm_min"]),
                "xiongjing": (tree["xiongjing"] - self.stats["xiongjing_min"])
                / (self.stats["xiongjing_max"] - self.stats["xiongjing_min"]),
                "guanfu": (tree["guanfu"] - self.stats["guanfu_min"])
                / (self.stats["guanfu_max"] - self.stats["guanfu_min"]),
                "x": (tree["geometry"].x - self.left) / self.width,
                "y": (tree["geometry"].y - self.bottom) / self.height,
            }
            assert np.any(obs.values() != "nan")
            return obs

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

            assert 0 <= height <= 1, f"Height out of range: {height}"
            assert 0 <= diameter <= 1, f"Diameter out of range: {diameter}"
            assert 0 <= crown_width <= 1, f"Crown width out of range: {crown_width}"
            assert 0 <= x <= 1, f"x coordinate out of range: {x}"
            assert 0 <= y <= 1, f"y coordinate out of range: {y}"

        obs = np.array(obs, dtype=np.float32)
        return obs

    def _calculate_reward(self, action):

        tree = self._get_single_observation(action)
        compete_index = self.fm.get_compete_index(action)
        self.fm.harvest_tree(action)

        # canopy_closure_penalty = -10 if self.fm.yubidu < CANOPY_CLOSURE_THRESHOLD else 0

        reward = (
            W1 * (1 - tree["max_chm"])
            + W2 * (1 - tree["xiongjing"])
            + W3 * (1 - tree["guanfu"])
            + W4 * self._sigmoid(compete_index, -1)
        )
        return reward

    def reset(self, seed=0, options=None):
        np.random.seed(seed)
        self.fm = ForestManager()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        reward = self._calculate_reward(action)
        done = self.fm.trees["is_cut"].sum() >= MAX_TREE_CUT
        obs = self._get_observation()
        return obs, reward, bool(done), False, {}

    def action_masks(self):
        return ~self.fm.trees["is_cut"].values

    def _sigmoid(self, x, k):
        return 1 / (1 + np.exp(k * x))


if __name__ == "__main__":
    env = TreeHarvestEnv()
    check_env(env, warn=True, skip_render_check=True)
