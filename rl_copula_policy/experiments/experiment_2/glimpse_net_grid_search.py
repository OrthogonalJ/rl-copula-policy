import os
import argparse
import logging

from sklearn.model_selection import ParameterGrid
import ray  # pylint: disable=import-error

from rl_copula_policy.policies.pg_policy import PGPolicy
from rl_copula_policy.experiments.experiment_utils import (
    current_timestamp, run_experiment_repeatedly, merge_configs)

# Importing to ensure action distributions, models and envs are
# registered with Ray's model catalog.
import rl_copula_policy.environments.catch_glimpse_env
import glimpse_network.models.glimpse_net_model
import glimpse_network.action_distributions.categorical_gaussian_diag_action_dist

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--export_dir',
                            default='./data/debug/experiment_2')
    args = arg_parser.parse_args()

    ray.init(logging_level=logging.INFO)

    EXP_NAME_PREFIX = 'glimpse_net_grid_search_'
    MODEL_NAME = 'glimpse_net_model'
    ENV_NAME = 'catch_glimpse'
    base_export_dir = '{}/glimpse_net_grid_search_{}'.format(
        args.export_dir, current_timestamp())
    os.makedirs(base_export_dir)

    seeds = [10, 21, 42]
    num_iter = 50
    # yapf: disable
    base_config = {
        'log_level': logging.INFO,
        'env': ENV_NAME,
        'num_gpus': 1,
        'num_workers': 16,
        'lr': 0.0005,
        'train_batch_size': 20000,
        'sample_batch_size': 1250,
        'gamma': 0.99,
        'eager': False,
        'env_config': {
            'max_balls': 1,
            'throw_rate': 10,
            'field_size': 8,
            'catcher_width': 1
        },
        'model': {
            'custom_model': MODEL_NAME,
            'custom_action_dist': 'categorical_gaussian_diag_action_dist',
            'use_lstm': True,
            'max_seq_len': 200
        }
    }
    base_config['model']['custom_options'] = {
        'obs_shape': [base_config['env_config']['field_size'], base_config['env_config']['field_size'], 1],
        'action_dim': 3,
        'n_patches': 1,
        'initial_glimpse_size': 4,
        # 'location_std': None,
        'sep_location_net_gradients': False,
        'baseline_input_type': 'image',
        # [action loss coef, location loss coef]
        # 'policy_loss_logp_coefs': [1.0, 1.0],
        # 'glimpse_net_activation': 'tanh',
        'vf_loss_coef': 1.0
    }

    param_grid_spec = {
        'model.custom_options.location_std': [None, 0.3, 0.03, 0.003],
        'model.custom_options.glimpse_net_activation': ['tanh', 'relu']
    }
    param_grid = list(ParameterGrid(param_grid_spec))
    # yapf:enable

    for params in param_grid:
        exp_config = merge_configs(base_config, params)
        exp_name = '{}_{}_{}'.format(EXP_NAME_PREFIX, MODEL_NAME, ENV_NAME)
        export_dir = '{}/{}'.format(base_export_dir, exp_name)

        run_experiment_repeatedly(exp_name,
                                  PGCopulaTrainer,
                                  num_iter=num_iter,
                                  base_export_dir=export_dir,
                                  config=exp_config,
                                  seeds=seeds)
