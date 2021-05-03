import itertools
from typing import Callable

from gym.wrappers.time_limit import TimeLimit
import numpy as np
from tqdm import tqdm


def q_learning_control_epsilon_greedy(
    env: TimeLimit,
    stats: dict,
    num_episodes: int,
    policy: Callable,
    discount_factor: float = 1.0,
    learning_rate: float = 0.5,
    max_epsilon: float = 1.0,
    min_epsilon: float = 0.001,
    decay_rate: float = 0.00005
) -> np.ndarray:
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy
    policy while following an epsilon-greedy policy

    Args:
        env: OpenAI environment
        stats: Dictionary contains statistics about the experiment
        num_episodes: Number of episodes to run for
        policy: Function that returns an action according to a policy
        discount_factor: Gamma discount factor
        learning_rate: TD learning rate
        max_epsilon: Max epsilon value from where the decay starts
        min_epsilon: Min epsilon value until the decay lasts
        decay_rate: Rate of the exponential decay

    Returns:
        Q table with state-action values
    """

    num_wins = 0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_table = np.zeros((num_states, num_actions))
    epsilon = max_epsilon

    for i_episode in tqdm(range(num_episodes)):

        state = env.reset()

        for t in itertools.count():

            action = policy(env, q_table, state, epsilon)
            next_state, reward, done, info = env.step(action)

            stats["train/episode_rewards"][i_episode] += reward
            stats["train/episode_lengths"][i_episode] = t

            td_target = reward + discount_factor * np.max(q_table[next_state])
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            state = next_state

            if done:
                if reward == 1:
                    num_wins += 1
                break

        stats["train/epsilon"][i_episode] = epsilon

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i_episode)

        win_ratio = num_wins / (i_episode + 1)
        stats['train/win_ratio'][i_episode] = win_ratio

        if i_episode % 5000 == 0 and i_episode > 0:
            print(f'Current win ratio is {win_ratio}, epsilon: {epsilon}')

    return q_table
