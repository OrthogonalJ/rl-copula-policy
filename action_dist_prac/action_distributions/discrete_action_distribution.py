import gym.spaces as action_spaces
from ray.rllib.models import ModelCatalog # pylint: disable=import-error
from ray.rllib.models.tf.tf_action_dist import ActionDistribution # pylint: disable=import-error
from tensorflow_probability import distributions as tfd
import tensorflow as tf

class DiscreteActionDistribution(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        if not isinstance(action_space, action_spaces.Discrete):
            raise NotImplementedError
        return action_space.n

    def __init__(self, inputs, model):
        super(DiscreteActionDistribution, self).__init__(inputs, model)
        self._dist = tfd.Categorical(logits=self.inputs, validate_args=True, 
                allow_nan_stats=False)

    def sample(self):
        sample = self._dist.sample()
        self._last_sample_logp = self._dist.log_prob(sample)
        return sample

    def logp(self, action):
        action = tf.cast(action, tf.int32)
        return self._dist.log_prob(action)
    
    def sampled_action_logp(self):
        return self._last_sample_logp

    def entropy(self):
        return self._dist.entropy()

    def kl(self, other):
        """
        Args:
            other: another DiscreteActionDistribution instance
        Returns: KL-Divergence between this distribution and other
        """
        return self._dist.kl_divergence(other._dist)

ModelCatalog.register_custom_action_dist('discrete_action_distribution', DiscreteActionDistribution)