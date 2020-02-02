import datetime
import os
import copy
from collections import deque

from sklearn.model_selection import ParameterGrid
import ray # pylint: disable=import-error
from ray import tune # pylint: disable=import-error
from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger # pylint: disable=import-error

from rl_copula_policy.utils.ray_logger import RayLogger


def current_timestamp():
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

def run_experiment(exp_name, trainable, num_iter, export_dir, config,
        export_trace=False):
    if export_trace:
        config['output'] = export_dir

    results = tune.run(
        trainable,
        name=exp_name,
        stop={'training_iteration': num_iter},
        local_dir=export_dir,
        loggers=DEFAULT_LOGGERS + (RayLogger,),
        config=config,
        checkpoint_at_end=True
    )
    return results

def run_experiment_repeatedly(exp_name, trainable, num_iter, base_export_dir, 
        config, seeds, export_trace=False):
    if not os.path.isdir(base_export_dir):
        os.makedirs(base_export_dir)
        # Create empyty flag file to tell consumers that this directory contains
        # results for multiple seeds
        open(os.path.join(base_export_dir, '.MULTIPLE_SEEDS'), 'a').close()
    
    results = []
    for seed in seeds:
        current_name = '{}_seed{}'.format(exp_name, seed)
        current_export_dir = os.path.join(base_export_dir, current_name)

        current_config = copy.deepcopy(config)
        current_config['seed'] = seed

        exp_results = run_experiment(current_name, trainable, num_iter, 
                current_export_dir, current_config, export_trace)
        results.append(exp_results)
    
    return results


def nested_dict_put(root_dict, key, value):
    key_parts = deque(key.split('.'))
    dict_to_mutate = root_dict
    while len(key_parts) > 1:
        current_key = key_parts.popleft()

        if not current_key in dict_to_mutate:
            dict_to_mutate[current_key] = {}
        
        dict_to_mutate = dict_to_mutate[current_key]
    dict_to_mutate[key_parts.popleft()] = value

def merge_configs(base_config, new_config):
    merged_config = copy.deepcopy(base_config)
    for key_path, value in new_config.items():
        nested_dict_put(merged_config, key_path, value)
    return merged_config
