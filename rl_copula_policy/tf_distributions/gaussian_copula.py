import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from rl_copula_policy.utils.utils import shape_list

class GaussianCopula(tfd.MultivariateNormalTriL):
    def __init__(self, loc, scale_tril):
        """
        Args:
            loc(tensor)
            covariance_matrix(tensor)
        """
        # scale_mat = tf.matmul(scale_tril, scale_tril, transpose_b=True)
        # print_op = tf.compat.v1.print('scale_mat:', scale_mat)
        # with tf.control_dependencies([print_op]):
        #     diagonal = tf.ones(shape=shape_list(scale_mat)[:-1])
        # scale_tril = tf.linalg.set_diag(tf.linalg.cholesky(scale_mat), diagonal)
        self._loc = loc
        self._scale_tril = scale_tril
        super(GaussianCopula, self).__init__(loc, scale_tril, validate_args=True)

    def sample(self):
        sample = super(GaussianCopula, self).sample()
        print('Gaussian Copula sample:', sample)
        print('Gaussian Copula stddev:', self._stddev())
        #sample = sample / self._stddev()
        return self._standardize_value(sample)
     
    def cdf(self, value):
        # print('self._variances():', self._variances())
        # print('self._loc:', self._loc)
        #normal_dist = tfd.Normal(loc=self._loc, scale=self._variances())
        #cdf_vals = normal_dist.cdf(value)
        
        std_normal_value = self._standardize_value(value)
        #std_normal_value = (value - self._loc) / self._stddev()
        
        normal_cdf = tfb.NormalCDF()
        value_parts = tf.unstack(std_normal_value, axis=-1)
        cdf_vals = [normal_cdf.forward(value_part) for value_part in value_parts]
        cdf_vals = tf.concat(cdf_vals, axis=-1)
        
        # Ensure that output has same leading dims as value arg
        non_data_shape = shape_list(value)[:-1]
        num_components = shape_list(value)[-1]
        cdf_vals = tf.reshape(cdf_vals, non_data_shape + [num_components])

        return cdf_vals

    def _standardize_value(self, value):
        return (value - self._loc) / self._stddev()

    def _variances(self):
        # Reverse the cholesky decomposition
        cov_mat = tf.matmul(self._scale_tril, self._scale_tril, transpose_b=True)
        return tf.linalg.diag_part(cov_mat)
    
    def _stddev(self):
        return tf.math.sqrt(self._variances())

# class GaussianCopula(tfd.TransformedDistribution):
#     def __init__(self, loc, scale_tril):
#         """
#         Args:
#             loc(tensor)
#             covariance_matrix(tensor)
#         """
#         distribution = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril, validate_args=True)
#         super(GaussianCopula, self).__init__(
#             distribution=distribution, 
#             bijector=tfb.NormalCDF(),
#             validate_args=True,
#             name='GaussianCopula'
#         )
