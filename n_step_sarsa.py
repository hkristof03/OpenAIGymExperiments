import itertools
from typing import Callable

from gym.wrappers.time_limit import TimeLimit
import numpy as np
from tqdm import tqdm

from policies import epsilon_greedy_policy


def n_step_sarsa(
        env: TimeLimit,
        stats: dict,
        num_episodes: int,
        policy: Callable,
        discount_factor: float = 0.9,
        learning_rate: float = 0.8,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.001,
        decay_rate: float = 0.00005,
        n_steps: int = 5
):
    """
    N-step Sarsa for estimating Q (N-step bootstrapping).

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
        n_steps: Steps to bootstrap

    Returns:
        Q table with state-action values
    """

    num_wins = 0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_table = np.zeros((num_states, num_actions))
    epsilon = max_epsilon

    for i_episode in tqdm(range(num_episodes)):

        T = np.inf

        state = env.reset()

        action = policy(env, q_table, state, epsilon)

        actions = [action]
        states = [state]
        rewards = [0]

        for t in itertools.count():

            if t < T:
                next_state, reward, done, _ = env.step(action)

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

                    if reward == 1:
                        num_wins += 1

                else:
                    action = epsilon_greedy_policy(env, q_table,
                                                   state, epsilon)
                    actions.append(action)

                stats["episode_rewards"][i_episode] += reward
                stats["episode_lengths"][i_episode] = t

            # state tau being updated
            tau = t - n_steps + 1

            if tau >= 0:

                G = 0

                for i in range(tau + 1, min(tau + n_steps + 1, T + 1)):
                    G += np.power(discount_factor, i - tau - 1) * rewards[i]

                if tau + n_steps < T:
                    st = states[tau + n_steps]
                    act = actions[tau + n_steps]
                    G += np.power(discount_factor, n_steps) * q_table[st, act]

                # update Q values
                st = states[tau]
                act = actions[tau]
                q_table[st, act] += learning_rate * (G - q_table[st, act])

            if tau == T - 1:
                break

        stats["epsilon"][i_episode] = epsilon
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i_episode)

        if i_episode % 5000 == 0 and i_episode > 0:
            print(
                f"Game won {num_wins} times. Current win ratio is "
                f"{num_wins / i_episode}, epsilon: {epsilon}"
            )

    print(f"Win ratio: {round(num_wins / num_episodes, 5)}")

    return q_table












