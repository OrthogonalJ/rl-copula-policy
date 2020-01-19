import logging, datetime
import ray # pylint: disable=import-error
from ray import tune # pylint: disable=import-error
from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger # pylint: disable=import-error

from rl_copula_policy.policies.pg_trainer import PGTrainer

# Importing to ensure envs, models, and action distributions are added to tune registry
import rl_copula_policy.environments.catch_gym_env
import glimpse_network.glimpse_net_model

if __name__ == '__main__':
    ray.init()
    env_name = 'catch_gym_env'
    # model_name = 'mlp_model'
    model_name = 'glimpse_net_model'
    exp_name = 'glimse_net_debug_{}_{}'.format(model_name, env_name)
    export_dir = './data/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    num_iter = 2
    config = {
        'env': env_name,
        'num_gpus': 0,
        'num_workers': 1,#47,
        'lr': 0.0005,
        'train_batch_size': 10,#10011,
        'sample_batch_size': 10,#213,
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
        'sep_location_net_gradients': False
    }

    results = tune.run(
        PGTrainer,
        name=exp_name,
        stop={'training_iteration': num_iter},
        local_dir=export_dir,
        loggers=DEFAULT_LOGGERS,
        config=config
    )