import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

def placeholder_like(array):
    tensor_shape = [None] * len(array.shape)
    placeholder = tf.compat.v1.placeholder(shape=tensor_shape, dtype=tf.as_dtype(array.dtype))
    return placeholder

def shape_list(tensor):
    tensor_shape = tf.shape(tensor)
    return [tensor_shape[i] for i in range(tensor.get_shape().ndims)]

def make_mlp(input_layer, num_outputs, output_activation=None, 
        num_layers=1, layer_size=32, activation='relu', name_prefix='', output_name=None):
    
    hidden_out = input_layer
    for i in range(num_layers):
        layer_name = '{}fc{}'.format(name_prefix, i)
        hidden_out = Dense(layer_size, activation=activation, name=layer_name)(hidden_out)
    
    output_tensor = Dense(num_outputs, activation=output_activation, name=output_name)(hidden_out)
    return output_tensor

def reward_to_go(rewards, gamma):
    """
        Calculate discounted reward-to-go for one episode
    """
    trajectory_rewards = []
    for i in range(len(rewards)):
        num_steps_to_go = len(rewards) - i
        gamma_coefs = np.power(np.full(num_steps_to_go, gamma), np.arange(num_steps_to_go))
        reward = np.sum(gamma_coefs * rewards[i:])
        trajectory_rewards.append(reward)
    return trajectory_rewards

def slice_back(tensor, size, begin=None, name='concat'):
    begin = begin if begin is not None else [0] * len(size)
    size_len = len(size)
    begin_len = len(begin)
    # size_len = size.get_shape()[-1]
    # begin_len = begin.get_shape()[-1]
    assert begin is None or size_len == begin_len, 'size and begin must have the same length'
    shape = tensor.get_shape()
    num_unsliced_dims = len(shape) - size_len
    size_ = [-1] * num_unsliced_dims
    begin_ = [0] * num_unsliced_dims
    size_.extend(size)
    begin_.extend(begin)
    # size_ = tf.constant([-1] * num_unsliced_dims)
    # begin_ = tf.constant([0] * num_unsliced_dims)
    # size_ = tf.concat([size_, size], axis=-1)
    # begin_ = tf.concat([begin_, begin], axis=-1)
    return tf.slice(tensor, size_, begin_, name)

class ConstantFunctor:
    def __init__(self, value):
        self._value = value
    def __call__(self, *args, **kwargs):
        return self._value

