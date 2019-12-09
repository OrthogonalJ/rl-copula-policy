from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from rl_copula_policy.utils.utils import make_mlp

class RNNModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, 
            model_config, name):
        super(MLPModel, self).__init__(obs_space, action_space, num_outputs, 
                model_config, name)
        custom_options = model_config['custom_options']
        num_layers = custom_options['num_layers']
        layer_size = custom_options['layer_size']
        activation = custom_options['activation']

        obs_in = Input(shape=obs_space.shape, name='obs', dtype='float32')
        flat_action_params = make_mlp(obs_in, num_outputs, num_layers=num_layers, 
                layer_size=layer_size, activation=activation, name_prefix='policy_hidden_',
                output_name='policy_flat_action_prams')
        value_func_out = make_mlp(obs_in, 1, num_layers=num_layers, 
                layer_size=layer_size, activation=activation, name_prefix='vf_hidden_',
                output_name='vf_out')
        self.base_model = Model(obs_in, [flat_action_params, value_func_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()
        
    def forward(self, input_dict, state, seq_lens):
        flat_action_params, self._value_out = self.base_model(input_dict['obs'])
        return flat_action_params, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("rnn_model", RNNModel)