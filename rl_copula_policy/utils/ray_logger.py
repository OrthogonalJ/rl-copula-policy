import os
from collections import defaultdict
from ray.tune.logger import Logger, pretty_print
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE)
import tensorflow as tf_
tf = tf_.compat.v1

from rl_copula_policy.utils.utils import placeholder_like

class RayLogger(Logger):
    def _init(self):
        self._sess = tf.Session()
        self._file_writer = None
        self._get_file_writer()  # inits self._file_writer
        self._placeholders = defaultdict(dict)

    def _get_file_writer(self):
        if self._file_writer is None:
            log_dir = os.path.join(self.logdir, 'custom_tensorboard')
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            self._file_writer = tf.summary.FileWriter(log_dir)

        return self._file_writer

    def _build_summaries(self, summary_inputs, step):
        histogram_inputs = summary_inputs['histogram']
        for name, tensor in histogram_inputs.items():
            if name not in self._placeholders['histogram']:
                constant = placeholder_like(tensor)
                tf.summary.histogram(name, constant)
                self._placeholders['histogram'][name] = constant

        scalar_inputs = summary_inputs['scalar']
        for name, tensor in scalar_inputs.items():
            if name not in self._placeholders['scalar']:
                constant = placeholder_like(tensor)
                tf.summary.scalar(name, constant)
                self._placeholders['scalar'][name] = constant

    def _summary_feed_dict(self, summary_inputs):
        feed_dict = {}
        for summary_type, summary_dict in summary_inputs.items():
            for name, value in summary_dict.items():
                placeholder = self._placeholders[summary_type][name]
                feed_dict[placeholder] = value
        return feed_dict

    def on_result(self, result):
        inputs_key = 'tf_summary_inputs'
        if not tf_.executing_eagerly():
            return
        
        summary_inputs = result[inputs_key]
        file_writer = self._get_file_writer()

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._build_summaries(summary_inputs, step)

        summary_feed_dict = self._summary_feed_dict(summary_inputs)
        summary = self._sess.run(tf.summary.merge_all(), feed_dict=summary_feed_dict)
        file_writer.add_summary(summary, global_step=step)

        file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()
