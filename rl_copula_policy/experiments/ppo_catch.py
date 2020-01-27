import os
import argparse
import functools

import tensorflow as tf
# from tensorflow.keras.layers import LeakyReLU

from sklearn.model_selection import ParameterGrid
import ray # pylint: disable=import-error

from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG # pylint: disable=import-error

from rl_copula_policy.policies.pg_copula_trainer import PGCopulaTrainer
from rl_copula_policy.environments.gaus_copula_gym_env_wrapper import GausCopulaGymEnvWrapper
from rl_copula_policy.experiments.experiment_utils import current_timestamp, run_experiment_repeatedly, merge_configs

# importing to ensure action distributions and models are registered with Ray's model catalog
import rl_copula_policy.action_distributions.discrete_action_distribution
import rl_copula_policy.action_distributions.gaussian_copula_action_distribution
import rl_copula_policy.models.mlp_model
import rl_copula_policy.models.rnn_model
import rl_copula_policy.environments.catch_env

def merge_dicts(a, b):
    import copy
    output = copy.deepcopy(a)
    for k, v in b.items():
        output[k] = v
    return output

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--export_dir', default='./data/debug')
    args = arg_parser.parse_args()

    ray.init()
    model_name = 'mlp_model'
    # model_name = 'rnn_model'
    env_name = 'catch'
    exp_name = 'ppo_{}'.format(env_name)
    base_export_dir = '{}/ppo_catch_{}'.format(args.export_dir, current_timestamp())
    os.makedirs(base_export_dir)

    seeds = [10]
    #seeds = [10, 21, 42]
    num_iter = 100
    base_config = {
        'num_gpus': 0,
        'num_workers': 7,#47,
        'lr': 0.0005,
        'train_batch_size': 994,#10011,
        'sample_batch_size': 142,#213,
        'sgd_minibatch_size': 994,#10000,
        'num_sgd_iter': 1,
        'gamma': 0.99,
        'eager': False,
        'env': env_name,
        'env_config': {
            'max_balls': 1,
            'throw_rate': 10,
            'field_size': 8,
            'catcher_width': 1
        },
        'model': {
            'custom_model': model_name,
            'custom_options': {
                'num_layers': 2,
                'layer_size': 256,
                'activation': functools.partial(tf.nn.leaky_relu, alpha=0.1),
                'reward_to_go': True,
                'use_vf_adv': True
            }
        }
    }
    # base_config = merge_dicts(PPO_DEFAULT_CONFIG, base_config)

    run_experiment_repeatedly(exp_name, 'PPO', num_iter=num_iter, 
            base_export_dir=base_export_dir, config=base_config, seeds=seeds)