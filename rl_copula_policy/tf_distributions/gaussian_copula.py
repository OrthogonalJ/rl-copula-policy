import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from rl_copula_policy.utils.utils import shape_list

class GaussianCopula(tfd.MultivariateNormalTriL):
    
    # def __init__(self, loc, scale_inv_tril):
    #     """
    #     Args:
    #         loc(tensor): Means
    #         scale_tril(tensor): Lower triangular cholesky factor for the covariance matrix. 
    #             Note that the diagonal elements (latent marginal variances) are NOT IGNORED
    #             but all calculations behave as if the diagonals of the covariance matrix are 
    #             set to 1 (by standardising samples to have marginal loc 0 and stddev 1).
    #     """
    #     self._loc = loc

    #     scale_inv_tril = scale_tril
    #     scale = tf.matmul(scale_inv_tril, scale_inv_tril, transpose_b=True)
    #     self._scale_tril = tf.linalg.cholesky(scale)
        
    #     #self._scale_tril = scale_tril
    #     super(GaussianCopula, self).__init__(self._loc, self._scale_tril, validate_args=True)

    def __init__(self, loc, scale_tril):
        """
        Args:
            loc(tensor): Means
            scale_tril(tensor): Lower triangular cholesky factor for the covariance matrix. 
                Note that the diagonal elements (latent marginal variances) are NOT IGNORED
                but all calculations behave as if the diagonals of the covariance matrix are 
                set to 1 (by standardising samples to have marginal loc 0 and stddev 1).
        """
        self._loc = loc
        self._scale_tril = scale_tril
        super(GaussianCopula, self).__init__(loc, scale_tril, validate_args=True)

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        sample = super(GaussianCopula, self).sample(sample_shape, seed, name, **kwargs)
        return self._standardize_value(sample)
    
    def prob(self, std_value):
        cov_mat = self._standardized_cov_matrix()
        cov_mat_inv = tf.linalg.inv(cov_mat)
        cov_det = tf.linalg.det(cov_mat)
        n = self._num_dims()
        #n = tf.shape(self._loc)[-1]
        z = 1 / (tf.pow(2 * np.pi, tf.cast(n / 2, tf.float32)) * tf.math.sqrt(cov_det))
        # print('z:', z)

        # make shape [...leading dims, 1, n] (batch of row vectors)
        value_trans = tf.expand_dims(std_value, axis=-2)
        means_trans = tf.expand_dims(self._loc, axis=-2)
        deviations_trans = value_trans - means_trans
        # has shape [... leading dims, n, 1] (batch of column vectors)
        value_col_vec = tf.linalg.matrix_transpose(value_trans)
        #value_col_vec = tf.linalg.matrix_transpose(std_value)
        means = tf.linalg.matrix_transpose(means_trans)
        deviations = value_col_vec - means
        # print('deviations_trans: ', deviations_trans.shape)
        # print('cov_mat_inv:', cov_mat_inv.shape)
        # print('deviations:', deviations.shape)

        # print('deviations_trans * cov_mat_inv:', tf.matmul(deviations_trans, cov_mat_inv).shape)
        # print('(deviations_trans * cov_mat_inv)*deviations:', tf.matmul(tf.matmul(deviations_trans, cov_mat_inv), deviations).shape)

        exp_operand = -0.5 * tf.matmul(
                tf.matmul(deviations_trans, cov_mat_inv), deviations)
        # Make batch of scalars
        exp_operand = tf.squeeze(exp_operand)
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

if __name__ == '__main__':
    import math
    import random
    random.seed(42)
    np.random.seed(42)
    num_dim = 2
    num_samples = 10000
    loc = tf.random.uniform(shape=(1, num_dim), minval=0.0, maxval=100.0)
    loc = tf.tile(loc, [num_samples, 1])
    scale_flat = tf.random.uniform(shape=(1, int((num_dim * (num_dim + 1)) / 2),), minval=0.0, maxval=100.0)
    scale_flat = tf.tile(scale_flat, [num_samples, 1])
    scale_tril = tfp.math.fill_triangular(scale_flat)
    scale_tril = tf.linalg.set_diag(scale_tril, tf.math.exp(tf.linalg.diag_part(scale_tril)))
    
    # gaus_copula = tfd.MultivariateNormalTriL(loc, scale_tril)
    gaus_copula = GaussianCopula(loc, scale_tril)
    samples = tf.squeeze(gaus_copula.sample())
    samples_numpy = samples.numpy()
    cdfs = gaus_copula.cdf(samples).numpy()
    print('cdfs: (shape: {}) {}'.format(cdfs.shape, cdfs))
    probs = gaus_copula.prob(samples).numpy()
    # print('probs: (shape: {}) {}'.format(probs.shape, probs))

    import pandas as pd
    data = pd.DataFrame({
        'latent_0': samples_numpy[..., 0],
        'latent_1': samples_numpy[..., 1],
        'cdf_0': cdfs[..., 0],
        'cdf_1': cdfs[..., 1],
        'probs': probs
    })
    data.to_csv('local/gaussian_copula_debugging/samples_with_metrics.csv', index=False)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from rl_copula_policy.utils.pickle_utils import dump_pickle
    fig = plt.figure()
    ax = Axes3D(fig)
    x = samples_numpy[..., 0]
    y = samples_numpy[..., 1]
    z = probs
    surf = ax.plot_trisurf(x, y, z)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_angle1_plot.png')
    plt.show()
    ax.azim = 60
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_angle2_plot.png')
    plt.show()
    #dump_pickle(fig, 'gaussian_copula_cdf_angle1_plot.pkl')
    
