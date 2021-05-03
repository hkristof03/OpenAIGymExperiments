from typing import Callable
import itertools

from gym.wrappers.time_limit import TimeLimit
import numpy as np
from tqdm import tqdm


def sarsa(
    env: TimeLimit,
    stats: dict,
    num_episodes: int,
    policy: Callable,
    discount_factor: float = 1.0,
    learning_rate: float = 0.5,
    max_epsilon: float = 1.0,
    min_epsilon: float = 0.03,
    decay_rate: float = 0.00005
) -> np.ndarray:
    """
    SARSA algorithm: On-policy TD control. Finds an optimal Q state-action
    value function.

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
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping
        state -> action values.
        stats is an EpisodeStats object with two numpy arrays for
        episode_lengths and episode_rewards.
    """

    num_wins = 0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_table = np.zeros((num_states, num_actions))
    epsilon = max_epsilon

    for i_episode in tqdm(range(num_episodes)):

        state = env.reset()
        action = policy(env, q_table, state, epsilon)

        for t in itertools.count():

            next_state, reward, done, _ = env.step(action)
            next_action = policy(env, q_table, next_state, epsilon)

            if reward == 1:
                num_wins += 1

            stats['episode_rewards'][i_episode] += reward
            stats['episode_lengths'][i_episode] = t

            td_target = reward + discount_factor * q_table[
                next_state, next_action]
            td_delta = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_delta

            if done:
                break

            action = next_action
            state = next_state

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i_episode)

        if i_episode % 5000 == 0 and i_episode > 0:
            print(
                f"Current win ratio is {num_wins / i_episode}, "
                f"epsilon: {epsilon}"
            )

    print(f"Win ratio: {round(num_wins / num_episodes, 5)}")

    return q_table
