import gymnasium as gym
from gymnasium import spaces
import numpy as np

MAX_TREE_CUT = 1500
GRID_SIZE = 20


class GridState:
    def __init__(self, grid_data):
        self.trees = grid_data["trees"]

    def get_state(self):
        if not self.trees:
            return {
                "tree_features": np.empty((0, 2), dtype=np.float32),
                "positions": np.empty((0, 2), dtype=np.float32),
            }
        return {
            "tree_features": np.array(
                [[t["crown"], t["diameter"]] for t in self.trees]
            ),
            "positions": np.array([[t["x"], t["y"]] for t in self.trees]),
        }


class ActionSpace:
    def __init__(self, max_trees):
        if max_trees < 0:
            raise ValueError("max_trees must be greater than 0")
        self.action_space = spaces.MultiBinary(max_trees)

    def parse_action(self, action):
        return [bool(a) for a in action]


class RewardFunction:
    def __init__(self):
        self.weak_weight = 0.7
        self.density_weight = 0.3

        self.crown_weight = 0.6
        self.diameter_weight = 0.4

    def calculate_reward(self, state, action):
        # Convert to Numpy array
        action = np.array(action, dtype=bool)
        selected_mask = action

        if not selected_mask.any():
            return 0.0

        tree_features = state["tree_features"]
        positions = state["positions"]

        crowns = tree_features[:, 0]
        diameters = tree_features[:, 1]

        crown_mean = crowns.mean()
        crown_std = crowns.std() + 1e-8
        normalized_crown = (crowns - crown_mean) / crown_std

        diameter_mean = diameters.mean()
        diameter_std = diameters.std() + 1e-8
        normalized_diameter = (diameters - diameter_mean) / diameter_std

        weakness = (1 - normalized_crown) * self.crown_weight + (
            1 - normalized_diameter
        ) * self.diameter_weight
        print(weakness.shape)
        print(selected_mask.shape)
        selected_weakness = weakness[selected_mask].mean()

        remaining_mask = ~selected_mask
        if not remaining_mask.any():
            density_score = 1.0
        else:
            remaining_pos = positions[remaining_mask]
            n_remaining = len(remaining_pos)
            k = min(5, n_remaining - 1)

            if k > 0:
                dist_matrix = np.linalg.norm(
                    remaining_pos[:, np.newaxis, :] - remaining_pos[np.newaxis, :, :],
                    axis=2,
                )
                knn_distances = np.partition(dist_matrix, kth=k + 1, axis=1)[
                    :, 1 : k + 1
                ]
                avg_distances = knn_distances.mean(axis=1)

                density = 1 - (avg_distances / GRID_SIZE)
                density_score = density.mean()
            else:
                density_score = 1.0

        reward = (
            selected_weakness * self.weak_weight + density_score * self.density_weight
        )
        return float(reward)


class ForestLoggingEnv(gym.Env):
    def __init__(self, all_grids):
        super().__init__()

        # init grid system
        self.grids = [GridState(grid) for grid in all_grids]
        self.current_grid = 0

        max_trees = max(len(g.trees) for g in self.grids)
        self.action_space = ActionSpace(
            len(self.grids[self.current_grid].trees)
        ).action_space
        self.observation_space = spaces.Dict(
            {
                "tree_features": spaces.Box(low=0, high=23, shape=(max_trees, 2)),
                "positions": spaces.Box(low=0, high=260, shape=(max_trees, 2)),
            }
        )
        self.reward_calculator = RewardFunction()

    def reset(self):
        self.current_grid = 0
        self.action_space = ActionSpace(
            len(self.grids[self.current_grid].trees)
        ).action_space
        return self.grids[0].get_state()

    def step(self, action):
        current_state = self.grids[self.current_grid].get_state()
        reward = self.reward_calculator.calculate_reward(current_state, action)

        self.current_grid += 1
        done = self.current_grid >= len(self.grids)

        if not done:
            self.action_space = ActionSpace(
                len(self.grids[self.current_grid].trees)
            ).action_space

        next_state = self.grids[self.current_grid].get_state() if not done else None
        return next_state, reward, done, {}
