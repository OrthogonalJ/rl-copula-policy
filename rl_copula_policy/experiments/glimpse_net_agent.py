import logging
import datetime
import os
import ray # pylint: disable=import-error
from ray import tune # pylint: disable=import-error
from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger # pylint: disable=import-error

from rl_copula_policy.policies.pg_trainer import PGTrainer

# Importing to ensure envs, models, and action distributions are added to tune registry
import rl_copula_policy.environments.catch_gym_env
import glimpse_network.models.glimpse_net_model
import glimpse_network.action_distributions.categorical_gaussian_diag_action_dist

if __name__ == '__main__':
    ray.init(logging_level=logging.INFO)
    env_name = 'catch_gym_env'
    model_name = 'glimpse_net_model'
    exp_name = 'glimse_net_debug_{}_{}'.format(model_name, env_name)
    export_dir = './data/debug/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    os.makedirs(export_dir)
    num_iter = 100
    config = {
        'log_level': logging.INFO,
        'env': env_name,
        'num_gpus': 0,
        'num_workers': 7,#47,
        'lr': 0.0005,
        'train_batch_size': 3997,#10011,
        'sample_batch_size': 571,#213,
        'gamma': 0.99,
        'seed': 10,
        'eager': False,
        'env_config': {
            'max_balls': 1,
            'throw_rate': 10,
            'field_size': 8,
            'catcher_width': 1
        },
        'model': {
            'custom_model': model_name,
            'custom_action_dist': 'categorical_gaussian_diag_action_dist',
            'use_lstm': True,
            'max_seq_len': 200
        }
    }
    config['model']['custom_options'] = {
        'obs_shape': [config['env_config']['field_size'], config['env_config']['field_size'], 1],
        'action_dim': 3,
        'n_patches': 1,
        'initial_glimpse_size': 4,
        'location_std': None,
        'sep_location_net_gradients': False,
        # [action loss coef, location loss coef]
        'policy_loss_logp_coefs': [1.0, 0.7054802006488384]
    }

    results = tune.run(
        PGTrainer,
        name=exp_name,
        stop={'training_iteration': num_iter},
        local_dir=export_dir,
        loggers=DEFAULT_LOGGERS,
        config=config
    )