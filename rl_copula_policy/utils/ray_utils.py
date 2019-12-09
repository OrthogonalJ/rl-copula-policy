import numpy as np
import tensorflow as tf
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

def sample_batch_to_columnar_dict(sample_batch):
    columnar_dict = {}
    for key in sample_batch:
        columnar_dict[key] = np.stack(sample_batch[key])
    return columnar_dict

def make_seq_mask(seq_lens, values, is_stateful=False):
    if is_stateful:
        max_seq_len = tf.reduce_max(seq_lens)
        mask = tf.sequence_mask(seq_lens, max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(values, dtype=tf.bool)
    return mask