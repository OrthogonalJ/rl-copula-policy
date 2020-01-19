import logging
import tensorflow as tf
import tensorflow_probability as tfp

from rl_copula_policy.utils.utils import shape_list

class GaussianDiag(tfp.distributions.MultivariateNormalDiag):
    @staticmethod
    def from_flat_tensor(flat_params, exponentiate_scale=False):
        assert flat_params.get_shape().as_list()[-1] % 2 == 0
        num_components = int(flat_params.get_shape().as_list()[-1] / 2)
        # print('num_components: {}'.format(num_components))
        means = flat_params[..., :num_components]
        # print('means: {}'.format(means))
        scale_diag = flat_params[..., num_components:]
        if exponentiate_scale:
            scale_diag = tf.math.exp(scale_diag)
        # print('scale_diag:', scale_diag)
        return GaussianDiag(loc=means, scale_diag=scale_diag, 
                validate_args=True)
    
    def __init__(self, *args, **kwargs):
        super(GaussianDiag, self).__init__(*args, **kwargs)
    
    def num_params(self):
        return self.loc.get_shape().as_list()[-1] * 2

    def flat_sample_size(self):
        return self.loc.get_shape().as_list()[-1]

    def flat_to_event_shape(self, flat_sample):
        # print('flat_sample:', flat_sample)
        return flat_sample