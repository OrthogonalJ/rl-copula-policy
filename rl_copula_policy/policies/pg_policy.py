import numpy as np
import tensorflow as tf
from ray.rllib.policy.sample_batch import SampleBatch # pylint: disable=import-error
from ray.rllib.policy.tf_policy import ACTION_LOGP # pylint: disable=import-error
from ray.rllib.policy.tf_policy_template import build_tf_policy # pylint: disable=import-error
from ray.rllib.agents.trainer_template import build_trainer # pylint: disable=import-error
from ray.rllib.agents.trainer import with_common_config # pylint: disable=import-error
from ray.rllib.evaluation.postprocessing import Postprocessing # pylint: disable=import-error
from rl_copula_policy.utils.utils import reward_to_go, ConstantFunctor
from rl_copula_policy.utils.ray_utils import sample_batch_to_columnar_dict, make_seq_mask
from rl_copula_policy.utils.tf_summary_register_mixin import TFSummaryRegisterMixin

SAMPLE_BATCH_REWARD_TO_GO_KEY = 'REWARD_TO_GOS'

DEFAULT_CONFIG = with_common_config({
    'lr': 0.0005,
    'num_workers': 0,
    'use_pytorch': False
})

DEFAULT_CUSTOM_OPTIONS = {
    'reward_to_go': True,
    'use_vf_adv': True,
    'vf_loss_coef': 1.0
}

def get_custom_option(policy, key):
    custom_options = policy.config['model']['custom_options']
    if key in DEFAULT_CUSTOM_OPTIONS.keys():
        return custom_options.get(key, DEFAULT_CUSTOM_OPTIONS[key])
    return custom_options[key]

def policy_gradient_loss(policy, model, dist_class, train_batch):
    # print('COMPUTING PG POLICY LOSS')
    vf_loss_coef = get_custom_option(policy, 'vf_loss_coef')
    actions = train_batch[SampleBatch.ACTIONS]
    advantages = train_batch[Postprocessing.ADVANTAGES]
    q_values = train_batch[Postprocessing.VALUE_TARGETS]
    vf_preds = model.value_function()
    seq_lens = train_batch['seq_lens'] if 'seq_lens' in train_batch else tf.ones_like(advantages)

    flat_action_params, state = model.from_batch(train_batch)
    mask = make_seq_mask(seq_lens, advantages, is_stateful=not not state)

    action_dist = dist_class(flat_action_params, model)

    policy_loss = -action_dist.logp(actions) * advantages

    vf_loss = vf_loss_coef * tf.square(vf_preds - q_values)

    total_loss = policy_loss + vf_loss
    return tf.reduce_mean(tf.boolean_mask(total_loss, mask))

def postprocess_sample_batch(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Args:
        policy
        sample_batch(SampleBatch): must contain the following extra metrics:
            SampleBatch.VF_PREDS
        other_agent_batches
        episode
    """
    trajectory = sample_batch_to_columnar_dict(sample_batch)
    trajectory = add_advantages(policy, trajectory)
    trajectory[Postprocessing.VALUE_TARGETS] = trajectory[SAMPLE_BATCH_REWARD_TO_GO_KEY].copy().astype(np.float32)
    new_sample_batch = SampleBatch(trajectory)
    return new_sample_batch

def add_advantages(policy, trajectory):
    gamma = policy.config['gamma']
    rewards = trajectory[SampleBatch.REWARDS]
    trajectory[SAMPLE_BATCH_REWARD_TO_GO_KEY] = np.array(reward_to_go(rewards, gamma), dtype=np.float32)
    if get_custom_option(policy, 'reward_to_go'):
        rewards = trajectory[SAMPLE_BATCH_REWARD_TO_GO_KEY]
    
    if get_custom_option(policy, 'use_vf_adv'):
        vf_preds = trajectory[SampleBatch.VF_PREDS]
        rewards = (rewards - vf_preds)
    
    trajectory[Postprocessing.ADVANTAGES] = rewards.copy().astype(np.float32)
    return trajectory

def extra_action_fetches(policy):
    return { SampleBatch.VF_PREDS: policy.model.value_function() }

def stats(policy, train_batch):
    stats =  {
        'action_logp_min': tf.reduce_min(train_batch[ACTION_LOGP]),
        'action_logp_max': tf.reduce_max(train_batch[ACTION_LOGP]),
        'action_logp_mean': tf.reduce_mean(train_batch[ACTION_LOGP]),
    }
    return stats

PGPolicy = build_tf_policy(
    name='pg_policy',
    loss_fn=policy_gradient_loss,
    get_default_config=ConstantFunctor(DEFAULT_CONFIG),
    postprocess_fn=postprocess_sample_batch,
    extra_action_fetches_fn=extra_action_fetches,
    stats_fn=stats
)
