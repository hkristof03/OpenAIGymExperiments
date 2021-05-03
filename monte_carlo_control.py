from collections import defaultdict
from typing import Callable

from gym.wrappers.time_limit import TimeLimit
import numpy as np
from tqdm import tqdm
import gym

from policies import epsilon_greedy_policy


def monte_carlo_control_epsilon_greedy(
    env: TimeLimit,
    stats: dict,
    num_episodes: int,
    policy: Callable,
    discount_factor: float = 1.0,
    max_epsilon: float = 1.0,
    min_epsilon: float = 0.001,
    decay_rate: float = 0.00005
) -> np.ndarray:
    """
    Monte Carlo Control using Epsilon-Greedy policies. Finds the optimal
    state-action value function.

    Args:
        env: OpenAI gym environment
        stats: Dictionary contains statistics about the experiment
        num_episodes: Number of episodes to sample
        policy: Function that returns an action according to a policy
        discount_factor: Gamma discount factor
        max_epsilon: Max epsilon value from where the decay starts
        min_epsilon: Min epsilon value until the decay lasts
        decay_rate: Rate of the exponential decay

    Returns:
        The optimal sate-action value function.
    """

    num_wins = 0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_table = np.zeros((num_states, num_actions))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    epsilon = max_epsilon

    for i_episode in tqdm(range(num_episodes)):

        unique_state_action_pairs = set()
        state_action_pairs_in_episode = []
        rewards_in_episode = []
        done = False
        state = env.reset()
        t = 0

        while not done:

            action = policy(env, q_table, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            state_action_pairs_in_episode.append([state, action])
            unique_state_action_pairs.add((state, action))
            rewards_in_episode.append(reward)
            state = next_state

            stats['train/episode_rewards'][i_episode] += reward
            stats['train/episode_lengths'][i_episode] = t

            if reward == 1:
                num_wins += 1

            t += 1

        state_action_pairs_in_episode = np.array(
            state_action_pairs_in_episode).astype(int)
        # Find all (state, action) pairs we've visited in this episode

        for state_action in unique_state_action_pairs:
            first_occurrence_idx = np.where(
                state_action_pairs_in_episode == state_action)[0][0]
            # Sum up all the rewards since the first occurrence
            G = sum([r * (discount_factor ** i) for i, r in
                     enumerate(rewards_in_episode[first_occurrence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state_action] += G
            returns_count[state_action] += 1.0
            st, act = state_action
            q_table[st, act] = (
                    returns_sum[state_action] / returns_count[state_action]
            )

        stats['train/epsilon'][i_episode] = epsilon

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i_episode)

        win_ratio = num_wins / (i_episode + 1)
        stats['train/win_ratio'][i_episode] = win_ratio

        if i_episode % 5000 == 0 and i_episode > 0:
            print(f'Current win ratio is {win_ratio}, epsilon: {epsilon}')

        # The policy is improved implicitly by changing the q_table
    print(f'Win ratio: {round(num_wins / num_episodes, 5)}')

    return q_table

