import os
import argparse

from sklearn.model_selection import ParameterGrid
import ray # pylint: disable=import-error

from rl_copula_policy.policies.pg_copula_trainer import PGCopulaTrainer
from rl_copula_policy.environments.gaus_copula_gym_env_wrapper import GausCopulaGymEnvWrapper
from rl_copula_policy.experiments.experiment_utils import current_timestamp, run_experiment_repeatedly, merge_configs

# importing to ensure action distributions and models are registered with Ray's model catalog
import rl_copula_policy.action_distributions.discrete_action_distribution
import rl_copula_policy.action_distributions.gaussian_copula_action_distribution
import rl_copula_policy.models.mlp_model
import rl_copula_policy.models.rnn_model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--export_dir', default='./data/experiment_1')
    args = arg_parser.parse_args()

    ray.init()
    # env_name = 'ReversedAddition3-v0'
    model_name = 'rnn_model'
    base_export_dir = '{}/ppo_baseline_{}'.format(args.export_dir, current_timestamp())
    os.makedirs(base_export_dir)

    # exp_name = 'gaussian_copula_v3_{}_{}'.format(model_name, env_name)
    # export_dir = './data/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    seeds = [10, 21, 42]
    num_iter = 1
    base_config = {
        #'output': export_dir,
        #'output_compress_columns': [],
        'num_gpus': 0,
        'num_workers': 7,#47,
        'lr': 0.0005,
        'train_batch_size': 1001,#10011,
        'sample_batch_size': 143,#213,
        'sgd_minibatch_size': 1001,#10000,
        'num_sgd_iter': 1,
        'gamma': 0.99,
        #'seed': 10,
        'eager': False,
        # 'log_level': 'DEBUG',
        # 'env_config': {
        #     'env_name': env_name
        # },
        'model': {
            'use_lstm': True,
            'custom_model': model_name,
            'max_seq_len': 2000,
            # 'custom_action_dist': 'gaussian_copula_action_distribution',
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
        'env': ['Reverse-v0', 'Copy-v0', 'ReversedAddition3-v0']
        # 'env_config.env_name': ['Reverse-v0', 'Copy-v0', 'RepeatCopy-v0', 'DuplicatedInput-v0', 'ReversedAddition3-v0']
    }
    param_grid = list(ParameterGrid(param_grid_spec))

    for params in param_grid:
        exp_config = merge_configs(base_config, params)
        env_name = exp_config['env']
        exp_name = 'ppo_baseline_{}_{}'.format(model_name, env_name)
        export_dir = '{}/{}'.format(base_export_dir, exp_name)

        run_experiment_repeatedly(exp_name, 'PPO', num_iter=num_iter, 
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



# import logging, datetime
# import ray # pylint: disable=import-error
# from ray import tune # pylint: disable=import-error
# from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger # pylint: disable=import-error


# from rl_copula_policy.policies.pg_trainer import PGTrainer
# from rl_copula_policy.policies.pg_copula_trainer import PGCopulaTrainer
# from rl_copula_policy.environments.latent_variable_gym_env_wrapper import LatentVariableGymEnvWrapper
# from rl_copula_policy.environments.gaus_copula_gym_env_wrapper import GausCopulaGymEnvWrapper
# from rl_copula_policy.utils.ray_logger import RayLogger

# # importing to ensure action distributions and models are registered with Ray's model catalog
# import rl_copula_policy.action_distributions.discrete_action_distribution
# import rl_copula_policy.action_distributions.gaussian_copula_action_distribution
# import rl_copula_policy.models.mlp_model
# import rl_copula_policy.models.rnn_model

# if __name__ == '__main__':
#     ray.init()
#     env_name = 'ReversedAddition3-v0'
#     # model_name = 'mlp_model'
#     model_name = 'rnn_model'
#     exp_name = 'ppo_baseline_{}_{}'.format(model_name, env_name)
#     export_dir = './data/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
#     results = tune.run(
#         'PPO',
#         # PGCopulaTrainer,
#         name=exp_name,
#         stop={'training_iteration': 100},
#         local_dir=export_dir,
#         loggers=DEFAULT_LOGGERS + (RayLogger,),
#         config={
#             'env': env_name,
#             # 'env': GausCopulaGymEnvWrapper,
#             'output': export_dir,
#             'output_compress_columns': [],
#             'num_gpus': 0,
#             'num_workers': 47,
#             'lr': 0.0005,
#             'train_batch_size': 10011,
#             'sample_batch_size': 213,
#             'sgd_minibatch_size': 10000,
#             'num_sgd_iter': 1,
#             'gamma': 0.99,
#             'seed': 10,
#             'eager': False,
#             # 'log_level': 'DEBUG',
#             'env_config': {
#                 'env_name': env_name
#             },
#             'model': {
#                 'use_lstm': True,
#                 'custom_model': model_name,
#                 'max_seq_len': 200,
#                 # 'custom_action_dist': 'gaussian_copula_action_distribution',
#                 'custom_options': {
#                     'num_layers': 3,
#                     'layer_size': 64,
#                     'activation': 'relu',
#                     'reward_to_go': True,
#                     'use_vf_adv': True
#                 }
#             }
#         }
#     )
#     print(results.trial_dataframes)
#     log_df = results.dataframe()
#     print(log_df.head())
