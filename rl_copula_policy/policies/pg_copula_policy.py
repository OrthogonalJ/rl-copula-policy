import tensorflow as tf
from ray.rllib.policy.sample_batch import SampleBatch # pylint: disable=import-error
from ray.rllib.evaluation.postprocessing import Postprocessing # pylint: disable=import-error
from rl_copula_policy.policies.pg_policy import PGPolicy, stats as pg_stats, policy_gradient_loss as pg_loss
from rl_copula_policy.utils.ray_utils import make_seq_mask
from rl_copula_policy.utils.tf_summary_register_mixin import TFSummaryRegisterMixin

def compute_loss(policy, model, dist_class, train_batch):
    loss = pg_loss(policy, model, dist_class, train_batch)
    flat_action_params, state = model.from_batch(train_batch)
    action_dist = dist_class(flat_action_params, model)
    actions = train_batch[SampleBatch.ACTIONS]
    action_logps = action_dist.logp(actions)
    
    advantages = train_batch[Postprocessing.ADVANTAGES]
    seq_lens = train_batch['seq_lens'] if 'seq_lens' in train_batch else tf.ones_like(advantages)
    mask = make_seq_mask(seq_lens, advantages, is_stateful=not not state)

    reduce_mean_masked = lambda values: tf.reduce_mean(tf.boolean_mask(values, mask))

    # For stats
    policy._last_loss_latent_samples = action_dist.extract_latent_sample(actions, seperate_into_list=True)
    

    # For tensorboard (ray_logger)
    policy._register_histogram_summary('action_logp', action_logps)

    latent_cdfs = action_dist._latent_dist.cdf(action_dist.extract_latent_sample(actions))
    latent_cdfs = tf.unstack(latent_cdfs, axis=-1)
    for i, cdf_val in enumerate(latent_cdfs):
        policy._register_histogram_summary('copula_std_gaus_cdf_dim_{}'.format(i), cdf_val)
    
    for i, var in enumerate(action_dist.marginal_variances()):
        policy._register_histogram_summary('marginal_copula_variance_{}'.format(i), var)

    for pair, cov in action_dist.marginal_pair_covariances():
        policy._register_histogram_summary('marginal_copula_covariance_{}_{}'.format(pair[0], pair[1]), cov)

    action_logp_parts = action_dist.logp_parts(actions)
    policy._register_scalar_summary('latent_action_logp', reduce_mean_masked(action_logp_parts[0]))
    for i, logp in enumerate(action_logp_parts[1:]):
        policy._register_scalar_summary('maginal_action_logp_{}'.format(i), reduce_mean_masked(logp))
    
    return loss

def get_stats(policy, train_batch):
    stats = pg_stats(policy, train_batch)
    for i, latent_sample in enumerate(policy._last_loss_latent_samples):
        stats['latent_sample_part_{}_min'.format(i)] = tf.reduce_min(latent_sample)
        stats['latent_sample_part_{}_mean'.format(i)] = tf.reduce_mean(latent_sample)
        stats['latent_sample_part_{}_max'.format(i)] = tf.reduce_max(latent_sample)
    return stats

def setup_mixins(policy, obs_space, action_space, config):
    TFSummaryRegisterMixin.__init__(policy)

PGCopulaPolicy = PGPolicy.with_updates(
    loss_fn=compute_loss,
    stats_fn=get_stats,
    before_loss_init=setup_mixins,
    mixins=[TFSummaryRegisterMixin]
)