import os
from collections import defaultdict
from ray.tune.logger import Logger, pretty_print
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE)
#import tensorflow as tf

tf = None

class RayLogger(Logger):
    def _init(self):
        global tf
        if tf is None:
            import tensorflow as tf
            tf = tf.compat.v1  # setting this for regular TF logger
        # global tf
        # if tf is None:
        #     import tensorflow as tf
        #     tf = tf.compat.v2  # setting this for TF2.0
        # super(RayLogger, self).__init__(config, logdir, trial)
        
        self._sess = tf.Session()
        self._file_writer = None
        self._get_file_writer() # inits self._file_writer
        self._placeholders = defaultdict(dict)
    
    def _get_file_writer(self):
        if self._file_writer is None:
            log_dir = os.path.join(self.logdir, 'custom_tensorboard')
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            
            self._file_writer = tf.summary.FileWriter(log_dir)
            #self._file_writer = tf.summary.create_file_writer(log_dir)
        
        return self._file_writer

    def _build_summaries(self, summary_inputs, step):
        histogram_inputs = summary_inputs['histogram']
        for name, tensor in histogram_inputs.items():
            if name not in self._placeholders['histogram']:
                constant = self._make_placeholder_like(tensor)
                tf.summary.histogram(name, constant)
                self._placeholders['histogram'][name] = constant
            # tf.summary.histogram(name=name, data=tensor, step=step)
        
        scalar_inputs = summary_inputs['scalar']
        for name, tensor in scalar_inputs.items():
            if name not in self._placeholders['scalar']:
                constant = self._make_placeholder_like(tensor)
                tf.summary.scalar(name, constant)
                self._placeholders['scalar'][name] = constant

            #summary_value = [tf.Summary.Value(tag=name, simple_value=tensor)]
            #summary = tf.Summary(value=summary_value)
            #self._file_writer.add_summary(summary, step)
            
            #tf.summary.scalar(name=name, data=tensor, step=step)

    def _summary_feed_dict(self, summary_inputs):
        feed_dict = {}
        for summary_type, summary_dict in summary_inputs.items():
            for name, value in summary_dict.items():
                placeholder = self._placeholders[summary_type][name]
                feed_dict[placeholder] = value
        return feed_dict
    
    def _make_placeholder_like(self, tensor):
        tensor_shape = [None for _ in tensor.shape]
        constant = tf.placeholder(shape=tensor_shape, dtype=tf.as_dtype(tensor.dtype))
        return constant

    def on_result(self, result):
        # print('RayLogger starting log...')
        # pretty_print(result)
        summary_inputs = result['tf_summary_inputs']
        file_writer = self._get_file_writer()

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._build_summaries(summary_inputs, step)

        summary_feed_dict = self._summary_feed_dict(summary_inputs)
        summary = self._sess.run(tf.summary.merge_all(), feed_dict=summary_feed_dict)
        file_writer.add_summary(summary, global_step=step)

        # with tf.device('/CPU:0'):
        #     with tf.summary.record_if(True), file_writer.as_default():
        #         step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        #         self._build_summaries(summary_inputs, step)
        file_writer.flush()
        # print('RayLogger finished logging.')

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()