import itertools
import scipy as sp
import numpy as np
import gym.spaces as gym_spaces
from ray.rllib.models import ModelCatalog # pylint: disable=import-error
from ray.rllib.models.tf.tf_action_dist import ActionDistribution # pylint: disable=import-error
from ray.rllib.policy.policy import TupleActions # pylint: disable=import-error
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf

from rl_copula_policy.action_distributions.discrete_action_distribution import DiscreteActionDistribution
from rl_copula_policy.tf_distributions.gaussian_copula import GaussianCopula
from rl_copula_policy.tf_distributions.categorical import Categorical
from rl_copula_policy.tf_distributions.gaussian_diag import GaussianDiag
from rl_copula_policy.utils.utils import slice_back, shape_list
from rl_copula_policy.utils.covariance_matrix_utils import tfp_scale_tril

class CategoricalGaussianDiagActionDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space[0].n + (action_space[1].shape[0] * 2)
    
    def __init__(self, inputs, model):
        super(CategoricalGaussianDiagActionDist, self).__init__(inputs, model)

        num_categories = model.action_space[0].n
        categorical_logits = inputs[..., :num_categories]
        self._categorical_dist = Categorical.from_flat_tensor(categorical_logits)
        
        gaussian_params = inputs[..., num_categories: ]
        self._gaussian_dist = GaussianDiag.from_flat_tensor(gaussian_params, exponentiate_scale=True)

        self._distributions = [self._categorical_dist, self._gaussian_dist]
    
    def sample(self):
        samples = [d.sample() for d in self._distributions]
        # print('samples:', samples)
        self._last_sample_logp = self._logp(samples)
        return TupleActions(samples)

    def logp_parts(self, flat_action):
        action_parts = self._extract_action_parts(flat_action)
        return self._logp_parts(action_parts)
    
    def logp(self, flat_action):
        action_parts = self._extract_action_parts(flat_action)
        return self._logp(action_parts)
    
    def sampled_action_logp(self):
        # print('sampled_action_logp:', self._last_sample_logp)
        return self._last_sample_logp

    def entropy(self):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def _logp_parts(self, action_parts):
        return [d.log_prob(a) for (a, d) in zip(action_parts, self._distributions)]
    
    def _logp(self, action_parts):
        # print('action_parts:', action_parts)
        # print('self._logp_parts(action_parts):', self._logp_parts(action_parts))
        logp_parts = self._logp_parts(action_parts)
        total_logp = 0
        for term in logp_parts:
            total_logp += term
        return term
        #return tf.reduce_sum(tf.concat(self._logp_parts(action_parts), axis=-1), axis=-1)

    def _extract_action_parts(self, flat_action):
        sample_parts = []
        next_free_idx = 0
        for d in self._distributions:
            start_idx = next_free_idx
            next_free_idx += d.flat_sample_size()
            flat_sample = flat_action[..., start_idx:next_free_idx]
            shaped_sample = d.flat_to_event_shape(flat_sample)
            # print('Extracting action for distribution (dist: {}, start_idx: {}, next_free_idx: {}, flat_sample: {}, shaped_sample: {}' \
                # .format(d, start_idx, next_free_idx, flat_sample, shaped_sample))
            sample_parts.append(shaped_sample)
        return sample_parts

ModelCatalog.register_custom_action_dist('categorical_gaussian_diag_action_dist', CategoricalGaussianDiagActionDist)
