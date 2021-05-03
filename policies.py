import random

from gym.wrappers.time_limit import TimeLimit
import numpy as np


def epsilon_greedy_policy(
    env: TimeLimit,
    q_table: np.ndarray,
    state: int,
    epsilon: float
) -> int:
    """

    :param env:
    :param q_table:
    :param state:
    :param epsilon:
    :return:
    """
    exp_exp_tradeoff = random.uniform(0, 1)

    if exp_exp_tradeoff > epsilon:
        action = np.argmax(q_table[state, :])
    else:
        action = env.action_space.sample()

    return action
