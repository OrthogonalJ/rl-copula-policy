from ray.rllib.agents.trainer_template import build_trainer # pylint: disable=import-error
from rl_copula_policy.policies.pg_copula_policy import PGCopulaPolicy

def write_tensorboard_summaries(trainer, result):
    policy = trainer.get_policy()
    result['tf_summary_inputs'] = policy.get_tf_summary_inputs()

PGCopulaTrainer = build_trainer(
    name='PGCopulaTrainer', 
    default_policy=PGCopulaPolicy,
    after_train_result=write_tensorboard_summaries
)
