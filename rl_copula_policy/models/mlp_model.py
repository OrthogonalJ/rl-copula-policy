from ray.rllib.models.tf.tf_modelv2 import TFModelV2 # pylint: disable=import-error
from ray.rllib.models import ModelCatalog # pylint: disable=import-error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense # pylint: disable=import-error
from tensorflow.keras import Model # pylint: disable=import-error
from rl_copula_policy.utils.utils import make_mlp

class MLPModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, 
            model_config, name):
        super(MLPModel, self).__init__(obs_space, action_space, num_outputs, 
                model_config, name)
        # store action space for use in action distribution
        self.action_space = action_space

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
        # Note: self._value_out is initialised when forward is called
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("mlp_model", MLPModel)
