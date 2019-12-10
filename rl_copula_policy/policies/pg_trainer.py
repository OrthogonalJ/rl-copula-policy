import tensorflow as tf
from ray.rllib.agents.trainer_template import build_trainer # pylint: disable=import-error
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,  # pylint: disable=import-error
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE) 
from rl_copula_policy.policies.pg_policy import PGPolicy, get_custom_option

PGTrainer = build_trainer(
    name='pg_trainer', 
    default_policy=PGPolicy
)
