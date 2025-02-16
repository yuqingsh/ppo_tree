# test_forest_env.py
import pytest
import numpy as np
from envs_bp import (
    GridState,
    ActionSpace,
    RewardFunction,
    ForestLoggingEnv,
    MAX_TREE_CUT,
)


@pytest.fixture
def sample_grids():
    return [
        {
            "trees": [
                {"crown": 2.5, "diameter": 0.3, "x": 10, "y": 15},
                {"crown": 3.1, "diameter": 0.8, "x": 12, "y": 18},
            ]
        },
        {"trees": [{"crown": 0.8, "diameter": 0.3, "x": 35, "y": 22}]},
    ]


# --------------------------
# GridState 测试用例
# --------------------------
class TestGridState:
    def test_empty_grid(self):
        grid = GridState({"trees": []})
        state = grid.get_state()
        assert state["tree_features"].shape == (0, 2)
        assert state["positions"].shape == (0, 2)

    def test_normal_grid(self):
        grid = GridState(
            {
                "trees": [
                    {"crown": 2.5, "diameter": 0.3, "x": 10, "y": 15},
                    {"crown": 3.1, "diameter": 0.8, "x": 12, "y": 18},
                ]
            }
        )
        state = grid.get_state()
        assert state["tree_features"].shape == (2, 2)
        assert state["positions"].shape == (2, 2)


# --------------------------
# ActionSpace 测试用例
# --------------------------
class TestActionSpace:
    @pytest.mark.parametrize(
        "max_trees,  expected_shape", [(5, (5,)), (10, (10,)), (0, (0,))]  # 边界情况
    )
    def test_action_space_shape(self, max_trees, expected_shape):
        space = ActionSpace(max_trees).action_space
        assert space.shape == expected_shape

    @pytest.mark.parametrize(
        "action,  expected",
        [
            ([1, 0, 1], [True, False, True]),
            ([0, 0, 0], [False, False, False]),
            ([1], [True]),  # 单棵树情况
        ],
    )
    def test_parse_action(self, action, expected):
        result = ActionSpace(len(action)).parse_action(action)
        assert result == expected


# --------------------------
# RewardFunction 测试用例
# --------------------------
class TestRewardFunction:
    @pytest.fixture
    def rf(self):
        return RewardFunction()

    def test_no_cutting(self, rf):
        state = {
            "trees": [{"crown": 2.5, "diameter": 0.3}, {"crown": 3.1, "diameter": 0.8}]
        }
        assert rf.calculate_reward(state, [0, 0]) == 0

    def test_single_tree_cutting(self, rf):
        # 使用GridState生成规范state
        grid = GridState({"trees": [{"crown": 1.0, "diameter": 0.5, "x": 10, "y": 10}]})
        state = grid.get_state()  # 获取包含tree_features的标准state
        assert rf.calculate_reward(state, [1]) == pytest.approx(0.3)  # 0.7*0 + 0.3*1

    def test_k0_case(self, rf):
        """测试当k=0时的边界情况"""
        state = {
            "trees": [
                {"crown": 2.5, "diameter": 0.3, "x": 10, "y": 10},
                {"crown": 3.1, "diameter": 0.8, "x": 15, "y": 15},
            ],
            "positions": np.array([[10, 10], [15, 15]]),
        }
        action = [1, 1]
        reward = rf.calculate_reward(state, action)
        assert not np.isnan(reward)


# --------------------------
# ForestLoggingEnv 测试用例
# --------------------------
class TestForestLoggingEnv:
    def test_reset(self, sample_grids):
        env = ForestLoggingEnv(sample_grids)
        state = env.reset()
        assert env.current_grid == 0
        assert state["tree_features"].shape == (2, 2)  # 第一个网格有2棵树

    def test_step_transition(self, sample_grids):
        env = ForestLoggingEnv(sample_grids)
        env.reset()
        _, _, done, _ = env.step([1, 1])
        assert env.current_grid == 1
        assert not done

        _, _, done, _ = env.step([1])
        assert done

    def test_observation_space(self, sample_grids):
        env = ForestLoggingEnv(sample_grids)
        assert env.observation_space["tree_features"].shape == (2, 2)
        assert env.observation_space["positions"].shape == (2, 2)

    def test_max_trees_calculation(self):
        test_grids = [{"trees": [{}] * 3}, {"trees": [{}] * 5}]
        env = ForestLoggingEnv(test_grids)
        assert env.action_space.shape == (5,)


# --------------------------
# 集成测试
# --------------------------
def test_full_episode(sample_grids):
    env = ForestLoggingEnv(sample_grids)
    state = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    assert isinstance(total_reward, float)
    assert 0 <= total_reward <= MAX_TREE_CUT
