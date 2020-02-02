import tensorflow as tf
import tensorflow_probability as tfp
from rl_copula_policy.utils.utils import shape_list

class Categorical(tfp.distributions.Categorical):
    @staticmethod
    def from_flat_tensor(flat_params):
        return Categorical(logits=flat_params)

    def log_prob(self, k):
        if self.logits_parameter().get_shape().ndims > 1 and (k.get_shape().ndims != (self.logits_parameter().get_shape().ndims - 1)):
            batch_shape = shape_list(self.logits_parameter())[:-1]
            k = tf.reshape(k, batch_shape)
        print('k:', k)
        print('logits_parameter:', self.logits_parameter())
        # NOTE: Default implementation seems to cause OOM errors when used with large batch sizes
        labels = tf.cast(k, tf.int32)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_parameter(), labels=labels)

    def num_params(self):
        return self._num_classes()

    def flat_sample_size(self):
        return 1

    # def quantile(self, value):
    #     num_classes = self._num_classes()
    #     class_cdfs = [self.cdf(tf.fill(tf.shape(value), k))
    #             for k in range(num_classes)]
    #     class_cdfs = tf.stack(class_cdfs, axis=-1)
    #     quantile = tf.math.argmax(tf.cast(class_cdfs >= value, tf.int32))
    #     return quantile

    def quantile(self, quantile):
        num_classes = self._num_classes()
        def next_class_fn(quantile, last_class):
            """
            Returns: first class greater than last_class with CDF value >= quantile
            """
            last_class_vec = tf.fill(tf.shape(quantile), last_class)
            if last_class == num_classes - 1:
                return last_class_vec
            
            current_class = last_class + 1
            current_class_vec = tf.fill(tf.shape(quantile), current_class)
            cdf = self.cdf(last_class_vec)
            return tf.where(cdf >= quantile,
                    last_class_vec,
                    next_class_fn(quantile, current_class))
        return next_class_fn(quantile, last_class=0)
    
    def flat_to_event_shape(self, flat_sample):
        return flat_sample

    def _num_classes(self):
        num_classes = (self.logits if self.logits is not None else self.probs) \
            .get_shape().as_list()[-1]
        return num_classes


if __name__ == '__main__':
    import numpy as np
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    sess = tf.Session()
    logits = tf.constant(np.log([0.25, 0.25, 0.25, 0.25]))
    categorical = Categorical(logits=logits)
    print('prob_params:', sess.run(categorical.probs_parameter()))
    for i in range(4):
        print('-' * 10)
        print('Class {}'.format(i))
        cdf_val = sess.run(categorical.cdf(i))
        print('CDF:', cdf_val)
        quantile = sess.run(categorical.quantile(cdf_val))
        print('quantile:', quantile)
        print('-' * 10)
    
