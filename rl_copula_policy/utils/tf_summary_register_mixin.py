import tensorflow as tf

class TFSummaryRegisterMixin:
    def __init__(self):
        self._tf_summary_input_register = {
            # each has format {name: tensor}
            'histogram': {},
            'scalar': {}
        }

    def get_tf_summary_inputs(self):
        return self._tf_summary_input_register
    
    def _register_tf_summary_input(self, name, tensor, summary_type):
        if not tf.executing_eagerly():
            return
        
        # inputs_map_for_type = self._tf_summary_input_register[summary_type]
        #if name in inputs_map_for_type:
            # print('WARNING: TF summary input for type {} with name {} already exists and will overwritten'.format(summary_type, name))
        self._tf_summary_input_register[summary_type][name] = tensor.numpy()

    def _register_histogram_summary(self, name, tensor):
        self._register_tf_summary_input(name, tensor, summary_type='histogram')

    def _register_scalar_summary(self, name, tensor):
        self._register_tf_summary_input(name, tensor, summary_type='scalar')

    def _build_summaries(self, step):
        histogram_inputs = self._tf_summary_input_register['histogram']
        for name, tensor in histogram_inputs.items():
            tf.summary.histogram(name=name, data=tensor, step=step)
        
        scalar_inputs = self._tf_summary_input_register['scalar']
        for name, tensor in scalar_inputs.items():
            tf.summary.scalar(name=name, data=tensor, step=step)