import tensorflow as tf
import tensorflow_probability as tfp

#### PARAMETERISATIONS ####

def tfp_scale_tril(flat_scale, **kwargs):
    """
    Opinionated TF approach to creating lower triangular cholesky factor for a symmetric positive definite matrix
    """
    kwargs['validate_args'] = kwargs.get('validate_args', True)
    return tfp.bijectors.ScaleTriL(**kwargs).forward(flat_scale)

def unit_row_norm_pos_def_tril(flat_scale, **kwargs):
    """
    Lower triangular matrix that is guaranteed to a lower triangular cholesky factor for a symmetric positive definite matrix 
    by normalising to rows so they sum to 1.
    """
    kwargs['validate_args'] = kwargs.get('validate_args', True)
    return tfp.bijectors.CorrelationCholesky(**kwargs).forward(flat_scale)

def positive_def_scale_inv_triu(scale_inv_flat):
    """
    Args:
        scale_inv_flat(Tensor): A flat vector representation of the upper triangular cholesky factor.
            For a correlation matrix with n marginals, the last axis must have 1 + 2 + ... + n elements.
    
    Returns: A upper cholesky factor for an inverse correlation matrix (aka the precision matrix). 
        The resultant inverse correlation matrix and correlation matrix should be positive definite.

    References:
        Williams, P. 1996, 'Using Neural Networks to Model Conditional Multivariate Densities', 
            Neural Computation, vol. 8, no. 4, pp. 842-54, 
            <https://www.researchgate.net/publication/14575523_Using_Neural_Networks_to_Model_Conditional_Multivariate_Densities>.
    """
    scale_inv_triu = tfp.math.fill_triangular(scale_inv_flat, upper=True)
    # Exponentiating the diagonal makes it positive which makes the complete matrix 
    # (scale_inv_triu^T * scale_inv_triu) positive definite.
    diagonal = tf.math.exp(tf.linalg.diag_part(scale_inv_triu))
    scale_inv_triu = tf.linalg.set_diag(scale_inv_triu, diagonal)
    return scale_inv_triu