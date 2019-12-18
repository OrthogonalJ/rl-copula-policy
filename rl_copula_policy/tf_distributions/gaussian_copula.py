import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from rl_copula_policy.utils.utils import shape_list

class GaussianCopula(tfd.MultivariateNormalTriL):
    def __init__(self, loc, scale_tril):
        """
        Args:
            loc(tensor): Means
            scale_tril(tensor): Lower triangular cholesky factor for the covariance matrix. 
                Note that the diagonal elements (latent marginal variances) will be ignored. 
                All calculation behave as if the diagonals are set to 1.
        """
        self._loc = loc
        self._scale_tril = scale_tril
        super(GaussianCopula, self).__init__(loc, scale_tril, validate_args=True)

    def sample(self):
        sample = super(GaussianCopula, self).sample()
        return self._standardize_value(sample)

    def prob(self, std_value):
        cov_mat = self._standardized_cov_matrix()
        cov_mat_inv = tf.linalg.inv(cov_mat)
        cov_det = tf.linalg.det(cov_mat)
        n = self._num_dims()
        #n = tf.shape(self._loc)[-1]
        z = 1 / (tf.pow(2 * np.pi, tf.cast(n / 2, tf.float32)) * tf.math.sqrt(cov_det))

        # make shape [...leading dims, 1, n] (batch of row vectors)
        value_trans = tf.expand_dims(std_value, axis=-2)
        means_trans = tf.expand_dims(self._loc, axis=-2)
        deviations_trans = value_trans - means_trans
        # has shape [... leading dims, n, 1] (batch of column vectors)
        value_col_vec = tf.linalg.matrix_transpose(std_value)
        means = tf.linalg.matrix_transpose(means_trans)
        deviations = value_col_vec - means
        
        exp_operand = -0.5 * tf.matmul(
                tf.matmul(deviations_trans, cov_mat_inv), deviations)
        density = z * tf.math.exp(exp_operand)
        return density

    def log_prob(self, std_value):
        return tf.math.log(self.prob(std_value)) 

    def cdf(self, value):
        value = self._standardize_value(value)
        
        normal_cdf = tfb.NormalCDF()
        value_parts = tf.unstack(value, axis=-1)
        cdf_vals = [normal_cdf.forward(value_part) for value_part in value_parts]
        cdf_vals = tf.concat(cdf_vals, axis=-1)
        
        # Ensure that output has same leading dims as value arg
        non_data_shape = shape_list(value)[:-1]
        num_components = shape_list(value)[-1]
        cdf_vals = tf.reshape(cdf_vals, non_data_shape + [num_components])

        return cdf_vals

    def _num_dims(self):
        return tf.shape(self._loc)[-1]

    def _standardize_value(self, value):
        return (value - self._loc) / self._stddev()

    def _standardized_cov_matrix(self):
        cov_matrix = self._cov_matrix()
        diagonal = tf.ones(shape=shape_list(self._loc))
        std_cov_matrix = tf.linalg.set_diag(cov_matrix, diagonal)
        return std_cov_matrix
        
    def _cov_matrix(self):
        # Reverse the cholesky decomposition
        return tf.matmul(self._scale_tril, self._scale_tril, transpose_b=True)

    def _variances(self):
        #cov_mat = tf.matmul(self._scale_tril, self._scale_tril, transpose_b=True)
        return tf.linalg.diag_part(self._cov_matrix())
    
    def _stddev(self):
        return tf.math.sqrt(self._variances())

