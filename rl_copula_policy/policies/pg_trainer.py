import tensorflow as tf
from ray.rllib.agents.trainer_template import build_trainer # pylint: disable=import-error
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,  # pylint: disable=import-error
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE) 
from rl_copula_policy.policies.pg_policy import PGPolicy, get_custom_option

def get_file_writer(trainer):
    if not hasattr(trainer, '_file_writer'):
        #policy = trainer.get_policy()
        trainer._file_writer = tf.summary.create_file_writer(trainer.logdir)
    return trainer._file_writer

def write_tensorboard_summaries(trainer, result):
    policy = trainer.get_policy()
    result['tf_summary_inputs'] = policy.get_tf_summary_inputs()
    # file_writer = get_file_writer(trainer)
    # with tf.summary.record_if(True), file_writer.as_default():
    #     print('RESULT KEYS:', list(result.keys()))
    #     step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
    #     policy._build_summaries()
    # file_writer.flush()

PGTrainer = build_trainer(
    name='pg_trainer', 
    default_policy=PGPolicy, 
    #after_train_result=write_tensorboard_summaries
)
