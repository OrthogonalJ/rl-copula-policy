import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from action_dist_prac.utils.utils import shape_list

class GaussianCopula(tfd.MultivariateNormalTriL):
    def __init__(self, loc, scale_tril):
        """
        Args:
            loc(tensor)
            covariance_matrix(tensor)
        """
        scale_mat = tf.matmul(scale_tril, scale_tril, transpose_b=True)
        diagonal = tf.ones(shape=shape_list(scale_mat)[:-1])
        scale_tril = tf.linalg.set_diag(tf.linalg.cholesky(scale_mat), diagonal)
        super(GaussianCopula, self).__init__(loc , scale_tril, validate_args=True)
    
    def cdf(self, value):
        # print('#' * 80)
        # print('VALUE:', value)
        # print('#' * 80)
        # num_components = value.get_shape().as_list()[-1]
        normal_cdf = tfb.NormalCDF()
        value_parts = tf.unstack(value, axis=-1)
        cdf_vals = [normal_cdf.forward(value_part) for value_part in value_parts]
        cdf_vals = tf.concat(cdf_vals, axis=-1)
        
        # Ensure that output has same leading dims as value arg
        non_data_shape = shape_list(value)[:-1]
        num_components = shape_list(value)[-1]
        cdf_vals = tf.reshape(cdf_vals, non_data_shape + [num_components])

        return cdf_vals

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