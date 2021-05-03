import gym
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
from typing import Union

import policies
import q_learning
from n_step_sarsa import n_step_sarsa
import monte_carlo_control
from neptune_ml_logger import NeptuneLogger
from utils import (
    read_yaml, write_q_table_to_file, save_stats_and_parameters_to_file
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--path_config', required=True, type=str,
                        help='path to the experiment config file')
    args = parser.parse_args()

    return args


def run_experiment(
    config: dict,
    parameters: dict,
    path_config: str,
    iteration: Union[int, None] = None
) -> None:
    """

    :param config:
    :param parameters:
    :param path_config:
    :param iteration:
    :return:
    """
    env = gym.make(**config['env'])

    nl = NeptuneLogger(**config, source_files=[path_config])

    stats = {
        'train/epsilon': np.zeros(parameters['num_episodes']),
        'train/episode_lengths': np.zeros(parameters['num_episodes']),
        'train/episode_rewards': np.zeros(parameters['num_episodes']),
        'train/win_ratio': np.zeros(parameters['num_episodes'])
    }

    module, function = config['rl_algorithm']
    rl_algorithm = getattr(globals()[module], function)
    q_table = rl_algorithm(env, stats, **parameters)

    nl.log_metrics(stats)
    path_file = write_q_table_to_file(env, q_table, config['file_name'])
    nl.log_q_table(path_file)
    save_stats_and_parameters_to_file(stats, config, params, iteration)


if __name__ == '__main__':

    arguments = parse_args()
    pc = arguments.path_config
    exp_conf = read_yaml(pc)

    params = deepcopy(exp_conf['parameters'])
    params['policy'] = getattr(policies, params['policy'])

    parameters_to_iterate = exp_conf['parameters_to_iterate']

    if parameters_to_iterate:

        for key, value in parameters_to_iterate.items():
            for it, param in enumerate(tqdm(np.linspace(*value))):
                params.update({key: param})
                exp_conf['parameters'].update({key: param})
                run_experiment(exp_conf, params, pc, it)
    else:
        run_experiment(exp_conf, params, pc)
