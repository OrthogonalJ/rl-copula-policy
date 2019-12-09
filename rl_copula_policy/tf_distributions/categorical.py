import tensorflow as tf
import tensorflow_probability as tfp

class Categorical(tfp.distributions.Categorical):
    @staticmethod
    def from_flat_tensor(flat_params):
        return Categorical(logits=flat_params)

    def num_params(self):
        return self._num_classes()

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
            cdf = self.cdf(current_class_vec)
            return tf.where(cdf >= quantile,
                    last_class_vec,
                    next_class_fn(quantile, current_class))
        return next_class_fn(quantile, last_class=0)
    
    def _num_classes(self):
        num_classes = (self.logits if self.logits is not None else self.probs) \
            .get_shape().as_list()[-1]
        return num_classes