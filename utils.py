import os
import yaml
from copy import deepcopy
from typing import Union

import pandas as pd
import numpy as np
from gym.wrappers.time_limit import TimeLimit


def read_yaml(path_file: str) -> dict:
    """Reads the experiment's config file.

    :param path_file: Location of the yaml config file
    :return: Experiment config in a dictionary
    """
    path_dir = os.path.dirname(__file__)

    with open(os.path.join(path_dir, 'experiments', path_file)) as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    return config


def write_q_table_to_file(
    env: TimeLimit,
    q_table: np.ndarray,
    file_name: str
) -> str:
    """The main purpose of the method is to log the Q table to neptune.ml. For
    this purpose unfortunately it must be written to a csv file and then
    the uploading of that file is possible.

    :param env: OpenAI Gym environment
    :param q_table: Q table resulted from an experiment
    :param file_name: Name of the csv file
    :return: Path to the csv file
    """
    n_actions = env.action_space.n
    columns = [f'action_{i}' for i in range(n_actions)]
    q_table_df = pd.DataFrame(q_table, columns=columns)
    path_file = os.path.join(
        os.path.dirname(__file__), 'artifacts', file_name + '.csv'
    )
    q_table_df.to_csv(path_file)

    return path_file


def save_stats_and_parameters_to_file(
    stats: dict,
    config: dict,
    params: dict,
    iteration: Union[int, None] = None
) -> None:
    """Main purpose of the function is to produce better plots with legends,
    etc. than neptune.ml, which misses some features. Therefore the
    experiment's parameters and the results are saved to two separate files.
    The name of the files are derived from the tags of the experiment.

    :param stats:
    :param config:
    :param params:
    :param iteration:
    :return:
    """
    tags = config['tags']
    name_stats = (
        [f'stats_iteration_{iteration}.csv']
        if iteration is not None else ['stats.csv']
    )
    name_params = (
        [f'params_iteration_{iteration}.csv']
        if iteration is not None else ['params.csv']
    )
    params = deepcopy(params)
    params.update({'policy': config['parameters']['policy']})
    derived_name_stats = '_'.join(tags + name_stats)
    derived_name_params = '_'.join(tags + name_params)
    path_artifacts = os.path.join(os.path.dirname(__file__), 'artifacts')
    path_stats = os.path.join(path_artifacts, derived_name_stats)
    path_params = os.path.join(path_artifacts, derived_name_params)
    pd.DataFrame(stats).to_csv(path_stats, index=False)
    pd.DataFrame(params, index=[0]).to_csv(path_params, index=False)






