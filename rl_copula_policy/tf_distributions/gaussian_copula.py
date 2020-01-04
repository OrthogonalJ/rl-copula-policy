import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from rl_copula_policy.utils.utils import shape_list

class GaussianCopula(tfd.MultivariateNormalTriL):
    
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
        return sample
    
    def prob(self, std_value):
        cov_mat = self._cov_matrix()
        cov_mat_inv = tf.linalg.inv(cov_mat)
        cov_det = tf.linalg.det(cov_mat)
        n = self._num_dims()
        z = 1 / (tf.pow(2 * np.pi, tf.cast(n / 2, tf.float32)) * tf.math.sqrt(cov_det))
        print('z:', z)

        # make shape [...leading dims, 1, n] (batch of row vectors)
        value_trans = tf.expand_dims(std_value, axis=-2)
        means_trans = tf.expand_dims(self._loc, axis=-2)
        deviations_trans = value_trans - means_trans
        # has shape [... leading dims, n, 1] (batch of column vectors)
        value_col_vec = tf.linalg.matrix_transpose(value_trans)
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
        marginal_normal_dist = tfd.Normal(loc=self._loc, scale=self._stddev())
        cdf_vals = marginal_normal_dist.cdf(value)
        
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
        return tf.linalg.diag_part(self._cov_matrix())
    
    def _stddev(self):
        return tf.math.sqrt(self._variances())

if __name__ == '__main__':
    import math
    import random
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    import seaborn as sns
    import scipy as sp
    from rl_copula_policy.utils.pickle_utils import dump_pickle
    # from rl_copula_policy.action_distributions.gaussian_copula_action_distribution import GaussianCopulaActionDistribution

    def vector_to_scale_inv_triu(vector):
        cholesky_factor = tfp.math.fill_triangular(vector, upper=True)
        # Ensure precision matrix (inverse scale/covariance matrix) is postive
        # definite by ensuring that diagonal is positive
        diagonal = tf.math.exp(tf.linalg.diag_part(cholesky_factor))
        cholesky_factor = tf.linalg.set_diag(cholesky_factor, diagonal)
        return cholesky_factor

    # A = tf.constant([[6, 2, 3], [2, 8, 4], [3, 4, 9]], dtype=tf.float32) #tf.random.uniform(shape=(3, 3), dtype=tf.float32)
    # print('A:', A)
    # A_inv = tf.linalg.inv(A)
    # print('A_inv:', A_inv)
    # A_inv_U = tf.linalg.matrix_transpose(tf.linalg.cholesky(A_inv))
    # print('A_inv_U:', A_inv_U)
    # A_inv_2 = tf.matmul(A_inv_U, A_inv_U, transpose_a=True)
    # print('A_inv_2:', A_inv_2)
    # A_2 = tf.linalg.inv(A_inv_2)
    # print('A_2:', A_2)

    # A_inv_L = tf.linalg.cholesky(A_inv)
    # print('A_inv_L:', A_inv_L)
    # A_L = tf.linalg.cholesky(A)
    # print('A_L:', A_L)
    # A_L2 = tfb.CholeskyToInvCholesky().forward(A_inv_L)
    # print('A_L2:', A_L2)

    # #print('tmp2:', tmp2)
    # exit()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        # TF V2
        tf.random.set_seed(SEED)
    except:
        # TF V1
        tf.random.set_random_seed(SEED)

    print(tf.executing_eagerly())

    num_dim = 2
    num_samples = 50000
    
    sess = tf.Session()
    sess.as_default().__enter__()

    loc = tf.zeros(shape=(1, num_dim), dtype=tf.float32)
    #loc = tf.random.uniform(shape=(1, num_dim), minval=-5.0, maxval=5.0)
    loc = tf.tile(loc, [num_samples, 1])
    scale_flat = tf.constant([[3.2912426, -1.1798649, -2.94235]], dtype=tf.float32)
    # scale_flat = tf.random.uniform(
    #     shape=(1, np.sum(np.arange(num_dim) + 1)), 
    #     minval=-10.0, maxval=10.0, dtype=tf.float32)
    scale_flat = tf.tile(scale_flat, [num_samples, 1])
    # print('scale_flat: (shape: {}){}'.format(scale_flat.numpy().shape, scale_flat.numpy()))
    print('scale_flat: (shape: {}){}'.format(sess.run(scale_flat).shape, sess.run(scale_flat)))
    # scale_tril = tfb.CorrelationCholesky().forward(scale_flat)
    # scale_triu = tfp.math.fill_triangular(scale_flat, upper=True)
    
    # scale_tril = tfb.CorrelationCholesky(validate_args=True).forward(scale_flat)
    scale_tril = tfb.ScaleTriL().forward(scale_flat)
    # scale_tril = tf.linalg.set_diag(scale_tril, tf.math.exp(tf.linalg.diag_part(scale_tril)))
    

    # print('scale_tril:', scale_tril)
    # print('scale:', tf.matmul(scale_tril, scale_tril, transpose_b=True))
    print('scale_tril:', sess.run(scale_tril))
    print('scale:', sess.run(tf.matmul(scale_tril, scale_tril, transpose_b=True)))
    
    gaus_copula = GaussianCopula(loc, scale_tril)
    # gaus_copula = GaussianCopula(loc, scale_triu, exponentiate_diag=True)

    # print('scale_flat:', scale_flat)
    # scale_tmp = tfb.CorrelationCholesky().forward(scale_flat)
    # print('scale_tmp:', scale_tmp.numpy())
    # A = tf.matmul(scale_tmp, scale_tmp, transpose_b=True)
    # print('A:', A)
    # print('tf.linalg.cholesky(A):', tf.linalg.cholesky(A))
    # A_diag = tf.linalg.diag_part(A)
    # print('A_diag:', A_diag)
    # print('loc:', loc)
    # normal_dist = tfd.Normal(loc=loc, scale=A_diag[..., 1:])
    # samples = gaus_copula.sample()
    # exit()

    # x, y = np.mgrid[-4:4:80j, -4:4:80j]
    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # grid = np.concatenate([x, y], axis=-1).astype(np.float32)
    # print('grid:', grid.shape)
    # grid_probs = gaus_copula.prob(grid)
    # print('grid_probs:', grid_probs.shape)
    # grid_cdfs = gaus_copula.cdf(grid)
    # print('grid_cdfs:', grid_cdfs.shape)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # x = grid[..., 0]
    # y = grid[..., 1]
    # z = grid_probs
    # surf = ax.plot_trisurf(x, y, z)
    # plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_grid_angle1_plot.png')

    # import pandas as pd
    # data = pd.DataFrame({
    #     'latent_0': grid[..., 0],
    #     'latent_1': grid[..., 1],
    #     'cdf_0': grid_cdfs[..., 0],
    #     'cdf_1': grid_cdfs[..., 1],
    #     'probs': grid_probs
    # })
    # data.to_csv('local/gaussian_copula_debugging/samples_with_metrics.csv', index=False)

    samples = tf.squeeze(gaus_copula.sample())
    # samples_numpy = sess.run(samples)
    # samples_numpy = samples.numpy()
    
    cdfs_tensor = gaus_copula.cdf(samples)
    # cdfs = gaus_copula.cdf(samples).numpy()
    
    probs_tensor = gaus_copula.prob(samples)
    # probs = gaus_copula.prob(samples).numpy()

    samples_numpy, cdfs, probs = sess.run((samples, cdfs_tensor, probs_tensor))
    print('cdfs: (shape: {}) {}'.format(cdfs.shape, cdfs))
    print('probs: (shape: {}) {}'.format(probs.shape, probs))


    data = pd.DataFrame({
        'latent_0': samples_numpy[..., 0],
        'latent_1': samples_numpy[..., 1],
        'cdf_0': cdfs[..., 0],
        'cdf_1': cdfs[..., 1],
        'probs': probs
    })
    data.to_csv('local/gaussian_copula_debugging/samples_with_metrics.csv', index=False)


    fig = plt.figure()
    ax = Axes3D(fig)
    x = samples_numpy[..., 0]
    y = samples_numpy[..., 1]
    z = probs
    surf = ax.plot_trisurf(x, y, z)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_probs_angle1_plot.png')
    plt.show()
    ax.azim = 60
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_probs_angle2_plot.png')
    plt.show()


    fig = plt.figure()
    ax = Axes3D(fig)
    x = cdfs[..., 0]
    y = cdfs[..., 1]
    z = probs
    surf = ax.plot_trisurf(x, y, z)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_vs_probs_angle1_plot.png')
    plt.show()
    ax.azim = 60
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_vs_probs_angle2_plot.png')
    plt.show()


    unvariate_cols = ['cdf_0', 'cdf_1', 'latent_0', 'latent_1']
    for col in unvariate_cols:
        plt.figure()
        sns.distplot(data.loc[:, col])
        plt.savefig('local/gaussian_copula_debugging/gaussian_copula_{}_plot.png'.format(col))
        plt.show()

    plt.figure()
    sns.jointplot(data.cdf_0, data.probs)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_0_vs_prob_plot.png')
    plt.show()

    plt.figure()
    sns.jointplot(data.cdf_1, data.probs)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_cdf_1_vs_prob_plot.png')
    plt.show()

    plt.figure()
    sns.jointplot(data.latent_0, data.latent_1)
    plt.savefig('local/gaussian_copula_debugging/gaussian_copula_latent_joint_plot.png')
    plt.show()