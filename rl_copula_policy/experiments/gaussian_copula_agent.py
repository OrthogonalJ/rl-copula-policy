import logging
import datetime
import os
import copy
from collections import deque

from sklearn.model_selection import ParameterGrid
import ray # pylint: disable=import-error
from ray import tune # pylint: disable=import-error
from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger, pretty_print # pylint: disable=import-error


from rl_copula_policy.policies.pg_trainer import PGTrainer
from rl_copula_policy.policies.pg_copula_trainer import PGCopulaTrainer
from rl_copula_policy.environments.latent_variable_gym_env_wrapper import LatentVariableGymEnvWrapper
from rl_copula_policy.environments.gaus_copula_gym_env_wrapper import GausCopulaGymEnvWrapper
from rl_copula_policy.utils.ray_logger import RayLogger

# importing to ensure action distributions and models are registered with Ray's model catalog
import rl_copula_policy.action_distributions.discrete_action_distribution
import rl_copula_policy.action_distributions.gaussian_copula_action_distribution
import rl_copula_policy.models.mlp_model
import rl_copula_policy.models.rnn_model

def current_timestamp():
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

def run_experiment(exp_name, trainable, num_iter, export_dir, config):
    results = tune.run(
        trainable,
        name=exp_name,
        stop={'training_iteration': num_iter},
        local_dir=export_dir,
        loggers=DEFAULT_LOGGERS + (RayLogger,),
        config=config
    )
    return results

def run_experiment_repeatedly(exp_name, trainable, num_iter, base_export_dir, config, seeds):
    if not os.path.isdir(base_export_dir):
        os.makedirs(base_export_dir)
        # Create empyty flag file to tell consumers that this directory contains
        # results for multiple seeds
        open(os.path.join(base_export_dir, '.MULTIPLE_SEEDS'), 'a').close()
    
    results = []
    for seed in seeds:
        current_config = copy.deepcopy(config)
        current_config['seed'] = seed
        current_name = '{}_seed{}'.format(exp_name, seed)
        current_export_dir = os.path.join(base_export_dir, current_name)
        exp_results = run_experiment(current_name, trainable, num_iter, 
                current_export_dir, current_config)
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


if __name__ == '__main__':
    ray.init()
    # env_name = 'ReversedAddition3-v0'
    model_name = 'rnn_model'
    base_export_dir = './data/gaussian_copula_v3_{}'.format(current_timestamp())
    os.makedirs(base_export_dir)

    # exp_name = 'gaussian_copula_v3_{}_{}'.format(model_name, env_name)
    # export_dir = './data/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    seeds = [10, 21, 42]
    num_iter = 200
    base_config = {
        'env': GausCopulaGymEnvWrapper,
        # 'output': export_dir,
        # 'output_compress_columns': [],
        'num_gpus': 0,
        'num_workers': 7,#47,
        'lr': 0.0005,
        'train_batch_size': 1001,#10011,
        'sample_batch_size': 143,#213,
        'gamma': 0.99,
        'eager': False,
        # 'env_config': {
        #     'env_name': env_name
        # },
        'model': {
            'custom_model': model_name,
            'max_seq_len': 2000,
            'custom_action_dist': 'gaussian_copula_action_distribution',
            'custom_options': {
                'num_layers': 3,
                'layer_size': 64,
                'activation': 'relu',
                'reward_to_go': True,
                'use_vf_adv': True
            }
        }
    }

    param_grid_spec = {
        'env_config.env_name': ['Reverse-v0', 'Copy-v0', 'ReversedAddition3-v0']
        # 'env_config.env_name': ['Reverse-v0', 'Copy-v0', 'RepeatCopy-v0', 'DuplicatedInput-v0', 'ReversedAddition3-v0']
    }
    param_grid = list(ParameterGrid(param_grid_spec))

    for params in param_grid:
        exp_config = merge_configs(base_config, params)
        env_name = exp_config['env_config']['env_name']
        exp_name = 'gaussian_copula_v3_{}_{}'.format(model_name, env_name)
        export_dir = '{}/{}'.format(base_export_dir, exp_name)

        run_experiment_repeatedly(exp_name, PGCopulaTrainer, num_iter=num_iter, 
                base_export_dir=export_dir, config=exp_config, seeds=seeds)
    
    # run_experiment(exp_name, PGCopulaTrainer, num_iter=100, export_dir=export_dir, 
    #         config=config)
    
    # results = tune.run(
    #     PGCopulaTrainer,
    #     name=exp_name,
    #     stop={'training_iteration': 100},
    #     local_dir=export_dir,
    #     loggers=DEFAULT_LOGGERS + (RayLogger,),
    #     config=config
    # )
    # print(results.trial_dataframes)
    # log_df = results.dataframe()
    # print(log_df.head())
