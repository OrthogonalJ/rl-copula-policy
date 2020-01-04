import logging, datetime
import ray # pylint: disable=import-error
from ray import tune # pylint: disable=import-error
from ray.tune.logger import DEFAULT_LOGGERS, JsonLogger, CSVLogger # pylint: disable=import-error


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

if __name__ == '__main__':
    ray.init()
    env_name = 'ReversedAddition3-v0'
    # model_name = 'mlp_model'
    model_name = 'rnn_model'
    exp_name = 'pg_baseline_{}_{}'.format(model_name, env_name)
    export_dir = './data/{}-{}'.format(exp_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    results = tune.run(
        'PG',
        # PGCopulaTrainer,
        name=exp_name,
        stop={'training_iteration': 100},
        local_dir=export_dir,
        loggers=DEFAULT_LOGGERS + (RayLogger,),
        config={
            'env': env_name,
            # 'env': GausCopulaGymEnvWrapper,
            'output': export_dir,
            'output_compress_columns': [],
            'num_gpus': 0,
            'num_workers': 7,
            'lr': 0.0005,
            'train_batch_size': 10003,
            'sample_batch_size': 1429,
            'gamma': 0.99,
            'seed': 10,
            'eager': False,
            'env_config': {
                'env_name': env_name
            },
            'model': {
                'custom_model': model_name,
                'max_seq_len': 20,
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
    )
    print(results.trial_dataframes)
    log_df = results.dataframe()
    print(log_df.head())
