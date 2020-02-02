import os
import argparse

from sklearn.model_selection import ParameterGrid
import ray  # pylint: disable=import-error

from rl_copula_policy.policies.pg_copula_trainer import PGCopulaTrainer
from rl_copula_policy.policies.pg_trainer import PGTrainer
from rl_copula_policy.environments.gaus_copula_gym_env_wrapper import GausCopulaGymEnvWrapper
from rl_copula_policy.environments.catch_glimpse_env import CatchGlimpseEnv
from rl_copula_policy.experiments.experiment_utils import current_timestamp, run_experiment_repeatedly, merge_configs

# importing to ensure action distributions and models are registered with Ray's model catalog
import rl_copula_policy.action_distributions.discrete_action_distribution
import rl_copula_policy.action_distributions.gaussian_copula_action_distribution
import rl_copula_policy.models.mlp_model
import rl_copula_policy.models.rnn_model
import glimpse_network.action_distributions.categorical_gaussian_diag_action_dist
import glimpse_network.models.glimpse_net_model

import logging
import pickle
import os
import glob
import time

import numpy as np

# CHECKPOINT_DIR = 'data/debug/gaussian_copula_debug/gaussian_copula_v3_20200201T173617/gaussian_copula_v3_rnn_model_ReversedAddition3-v0/gaussian_copula_v3_rnn_model_ReversedAddition3-v0_seed10/gaussian_copula_v3_rnn_model_ReversedAddition3-v0_seed10/PGCopulaTrainer_GausCopulaGymEnvWrapper_26709630_2020-02-01_17-36-17zpy_tlrg/checkpoint_10'
# CHECKPOINT_DIR = 'data/debug/glimse_net_debug_glimpse_net_model_catch_glimpse-20200201T213122/glimse_net_debug_glimpse_net_model_catch_glimpse/pg_trainer_catch_glimpse_fe1a0dda_2020-02-01_21-31-228wv0bmlq/checkpoint_10'
DEFAULT_CHECKPOINT = 'data/debug/glimse_net_debug_glimpse_net_model_catch_glimpse-20200202T011034/glimse_net_debug_glimpse_net_model_catch_glimpse/pg_trainer_catch_glimpse_9d26a488_2020-02-02_01-10-3449bs5yuj/checkpoint_10/checkpoint_10'

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--checkpoint', default=DEFAULT_CHECKPOINT)
arg_parser.add_argument('-e', '--episodes', default=10, type=int)
arg_parser.add_argument('-o', '--output_file', default='output.wmv')
args = arg_parser.parse_args()

num_episodes = args.episodes
checkpoint_file = args.checkpoint
checkpoint_dir = os.path.dirname(checkpoint_file)
output_file = args.output_file

ray.init(logging_level=logging.ERROR, local_mode=True)

config_file = os.path.join(checkpoint_dir, '..', 'params.pkl')
with open(config_file, 'rb') as fd:
    config = pickle.load(fd)
config['log_level'] = logging.ERROR
config['num_workers'] = 0

trainer = PGTrainer(env=config['env'], config=config)
trainer.restore(checkpoint_file)
# trainer = PGCopulaTrainer(env=config['env'], config=config)
policy = trainer.get_policy()
model = trainer.get_policy().model

env = CatchGlimpseEnv(config['env_config'])
# env = GausCopulaGymEnvWrapper(config['env_config'])
from ray.rllib.models.preprocessors import get_preprocessor
prep = get_preprocessor(env.observation_space)(env.observation_space)

# time.sleep(30)

# import cv2
# width = env._env._field_size
# height = env._env._field_size
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# #fourcc = cv2.VideoWriter_fourcc('M','R','L','E')
# fps = 20
# video_filename = 'output.avi'
# out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

import imageio
out = imageio.get_writer(output_file, mode='I', fps=1, quality=10)
# out = imageio.get_writer('output.gif', mode='I', fps=1, loop=1)

for _ in range(num_episodes):
    print('Episode {}'.format(_))
    print('-' * 20)
    obs = env.reset()
    done = False
    model_state = model.get_initial_state()
    action = None
    reward = None
    info = None
    trainer_info = None
    while not done:
        try:
            frame = env.render(info=trainer_info)
            if frame is not None:
                out.append_data(frame)
                # out.write(frame)
                # time.sleep(1)
        except Exception as e:
            print(e)
        action, model_state, trainer_info = trainer.compute_action(
            observation=obs, state=model_state, info=info, full_fetch=True)
        action = [a[0] for a in action]
        # model_state = [s[0] for s in model_state]
        obs, reward, done, info = env.step(action)

out.close()
# out.release()

# if __name__ == '__main__':
#     ray.init()
#     model_name = 'rnn_model'

#     seeds = [10]
#     num_iter = 10
#     env_name = 'ReversedAddition3-v0'
#     base_config = {
#         # 'env': GausCopulaGymEnvWrapper,
#         'num_gpus': 0,
#         'num_workers': 7,#47,
#         'lr': 0.0005,
#         'train_batch_size': 1001,#10011,
#         'sample_batch_size': 143,#213,
#         'gamma': 0.99,
#         'eager': False,
#         'env_config': {
#             'env_name': env_name
#         },
#         'model': {
#             'custom_model': model_name,
#             'max_seq_len': 2000,
#             'custom_action_dist': 'gaussian_copula_action_distribution',
#             'custom_options': {
#                 'num_layers': 3,
#                 'layer_size': 64,
#                 'activation': 'relu',
#                 'reward_to_go': True,
#                 'use_vf_adv': True
#             }
#         }
#     }

# trainer = PGCopulaTrainer(env=GausCopulaGymEnvWrapper, config=base_config)
# policy = trainer.get_policy()
# model = policy.model

# for i in range(num_iter):
#     trainer.train()

# env = GausCopulaGymEnvWrapper(base_config['env_config'])
# obs = env.reset()
# action = trainer.compute_action(observation=obs, state=model.get_initial_state(), full_fetch=True)
# print(action)
