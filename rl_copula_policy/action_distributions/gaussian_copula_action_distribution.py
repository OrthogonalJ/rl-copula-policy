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
from rl_copula_policy.utils.utils import slice_back, shape_list
from rl_copula_policy.utils.covariance_matrix_utils import tfp_scale_tril

# Constants
MARGINAL_ENV_ACTION_SPACE_START_IDX = 2

class MarginalSpec:
    def __init__(self, dist_class, num_params):
        self.dist_class = dist_class
        self.num_params = num_params

def choose_marginal_spec_for_action_space(action_space):
    if isinstance(action_space, gym_spaces.Discrete):
        return MarginalSpec(Categorical, action_space.n)
    raise NotImplementedError

class GaussianCopulaActionDistribution(ActionDistribution):

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        if not isinstance(action_space, gym_spaces.Tuple):
            raise NotImplementedError('GaussianCopulaActionDistribution only supports Tuple action spaces')

        num_marginals = len(action_space) - MARGINAL_ENV_ACTION_SPACE_START_IDX
        #num_marginals = len(action_space) - 1
        num_marginal_params = 0
        for i in range(MARGINAL_ENV_ACTION_SPACE_START_IDX, len(action_space)):
            spec = choose_marginal_spec_for_action_space(action_space[i])
            num_marginal_params += spec.num_params
        
        num_latent_params = GaussianCopulaActionDistribution.num_covariance_params(num_marginals)
        return num_marginal_params + num_latent_params
    
    @staticmethod
    def num_covariance_params(num_marginals):
        # return sp.special.comb(num_marginals, 2, exact=True)
        return num_marginals + sp.special.comb(num_marginals, 2, exact=True)

    def __init__(self, inputs, model):
        super(GaussianCopulaActionDistribution, self).__init__(inputs, model)
        assert inputs.get_shape().ndims in [2, 3], 'inputs must be 2D or 3D'
        # print('inputs shape:', self.inputs.shape)
        # print('inputs:', inputs)

        # NOTE: expected format is [copula_cdf_space, copula_latent_space, marginals...]
        marginal_spaces = model.action_space[MARGINAL_ENV_ACTION_SPACE_START_IDX : ]
        self.num_marginals = len(marginal_spaces)
        
        self._init_latent_dist(self.inputs, next_unused_idx=0, 
                num_marginals=self.num_marginals)
        
        self._marginal_dists = self._make_marginal_dists(marginal_spaces, self.inputs, 
                next_unused_idx=self._marginal_input_start_idx())

        # hacky way of keeping track of indexes used to unpack flat action representation
        self._flat_action_latent_start = self.num_marginals
        self._flat_action_latent_end = self._flat_action_latent_start + self.num_marginals

        self._flat_action_marginals_start = self._flat_action_latent_end
        self._flat_action_marginals_end = self._flat_action_marginals_start + self.num_marginals
    
    def sample(self):
        latent_sample = self._latent_dist.sample() # shape [num_marginals]

        if self.inputs.get_shape().ndims > 2:
            # if recurrent
            # add batch and time dims
            latent_sample = tf.reshape(latent_sample, [1, 1, -1])
        else:
            # if not recurrent
            # add batch dim
            latent_sample = tf.reshape(latent_sample, [1, -1])

        marginal_cdf_vals = [latent_sample[..., i] for i in range(len(self._marginal_dists))] 
        marginal_samples = [dist.quantile(cdf_val) for cdf_val, dist 
                in zip(marginal_cdf_vals, self._marginal_dists)]
        
        # Ensuring that event dim is defined (otherwise self._latent_dist.cdf() fails when run 
        # in non-eager or eager-tracing mode)
        latent_sample = tf.reshape(latent_sample, shape_list(latent_sample)[:-1] + [self.num_marginals])
        copula_cdf_vals = self._latent_dist.cdf(latent_sample)
        # print('copula_cdf_vals shape:', copula_cdf_vals.shape)

        self._last_sample_logp = self._logp(latent_sample, marginal_samples)
        result = TupleActions([copula_cdf_vals, latent_sample] + marginal_samples)
        # print('Sampling result:', [copula_cdf_vals, latent_sample] + marginal_samples)
        return result

    def logp(self, action):
        latent_variable = self.extract_latent_sample(action)
        base_actions = self.extract_marginal_samples(action)
        logp = self._logp(latent_variable, base_actions)
        # print('logp:', logp.shape)
        return logp
    
    def logp_parts(self, action):
        latent_variable = self.extract_latent_sample(action)
        base_actions = self.extract_marginal_samples(action)
        return self._logp_parts(latent_variable, base_actions)
    
    def sampled_action_logp(self):
        # print('self._last_sample_logp:', self._last_sample_logp.shape)
        return self._last_sample_logp

    def entropy(self):
        raise NotImplementedError

    def extract_latent_sample(self, action, seperate_into_list=False):
        if seperate_into_list:
            return [action[..., i] for i in range(self._flat_action_latent_start, self._flat_action_latent_end)]
        return action[..., self._flat_action_latent_start:self._flat_action_latent_end]
        # last_dim_size = self._flat_action_latent_end - self._flat_action_latent_start
        # latent_sample_shape = shape_list(latent_sample)[:-1] + [last_dim_size]
        # latent_sample = tf.reshape(latent_sample, latent_sample_shape)
    
    def extract_marginal_samples(self, action):
        return [action[..., i] for i in range(self._flat_action_marginals_start, self._flat_action_marginals_end)]

    def covariance_matrix(self):
        return self._latent_dist.cov_matrix()
        #return tf.matmul(self._latent_covariance_tril, self._latent_covariance_tril, transpose_b=True)
    
    def marginal_variances(self, as_list=True):
        cov_mat = self.covariance_matrix()
        variances = [cov_mat[..., i, i] for i in range(self.num_marginals)]
        if as_list:
            return variances
        return tf.concat(variances, axis=-1)

    def marginal_pair_covariances(self):
        cov_mat = self.covariance_matrix()
        pairs = itertools.combinations(range(self.num_marginals), 2)
        return [( (i, j), cov_mat[..., i, j] ) for i, j in pairs]
    
    def _init_latent_dist(self, flat_action_params, next_unused_idx, num_marginals):
        batch_time_shape = shape_list(flat_action_params)[:-1]
        latent_means = tf.zeros(shape=batch_time_shape + [num_marginals], dtype=tf.float32)
        # print('latent_means.shape:', latent_means.shape)

        num_covar_params = GaussianCopulaActionDistribution.num_covariance_params(self.num_marginals)
        latent_covariance_vec = self._extract_params_from_flat_tensor(flat_action_params, num_covar_params, next_unused_idx)
        self._latent_covariance_tril = tfp_scale_tril(latent_covariance_vec)
        # print('self._latent_covariance_tril:', self._latent_covariance_tril)
        # self._latent_covariance_tril = tfp.math.fill_triangular(latent_covariance_vec, name='latent_covariance_tril')
        
        self._latent_dist = GaussianCopula(latent_means, self._latent_covariance_tril)
    
    def _make_marginal_dists(self, action_space, flat_action_params, next_unused_idx=0):
        marginal_specs = [choose_marginal_spec_for_action_space(space)
                for space in action_space]
        marginal_dists = []
        for spec in marginal_specs:
            start_idx = next_unused_idx
            next_unused_idx += spec.num_params
            params = self._extract_params_from_flat_tensor(flat_action_params, spec.num_params, start_idx)
            dist = spec.dist_class.from_flat_tensor(params)
            marginal_dists.append(dist)
        return marginal_dists
    
    def _marginal_input_start_idx(self):
        return self.num_marginals

    def _logp(self, latent_variable, base_actions):
        latent_logp = self._latent_dist.log_prob(latent_variable)
        # print('_logp$latent_logp:', latent_logp.shape)
        total_marginal_logp = 0
        for action, dist in zip(base_actions, self._marginal_dists):
            total_marginal_logp += dist.log_prob(action)
        # print('_logp$total_marginal_logp:', total_marginal_logp.shape)
        total_logp = latent_logp + total_marginal_logp
        # print('_logp$total_logp:', total_logp.shape)
        return total_logp

    def _logp_parts(self, latent_variable, base_actions):
        latent_logp = self._latent_dist.log_prob(latent_variable)
        base_logps = [dist.log_prob(action) for action, dist in zip(base_actions, self._marginal_dists)]
        return [latent_logp] + base_logps
    
    def _extract_params_from_flat_tensor(self, flat_action_params, num_params, begin_idx=0):
        num_flat_rep_dims = flat_action_params.get_shape().ndims
        if num_flat_rep_dims > 2:
            # assuming flat_action_params has structure [batch, time, params]
            params = flat_action_params[:, :, begin_idx:(begin_idx + num_params)]
        else:
            # assuming flat_action_params has structure [batch, params]
            params = flat_action_params[:, begin_idx:(begin_idx + num_params)]
        return params
    
    def _num_marginal_params(self):
        return np.sum([spec.num_params() for spec in self._marginal_dists])

ModelCatalog.register_custom_action_dist('gaussian_copula_action_distribution', GaussianCopulaActionDistribution)
