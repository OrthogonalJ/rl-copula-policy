import numpy as np
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model
from rl_copula_policy.utils.utils import make_mlp

class RNNModel(RecurrentTFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, 
            model_config, name, hiddens_size=256, cell_size=64):
        super(RNNModel, self).__init__(obs_space, action_space, num_outputs, 
            model_config, name)
        self.cell_size = cell_size
        print('num_outputs:', num_outputs)
        custom_options = model_config['custom_options']
        num_layers = custom_options['num_layers']
        layer_size = custom_options['layer_size']
        activation = custom_options['activation']

        obs_in = Input(shape=(None, obs_space.shape[0]), name='obs', dtype='float32')
        state_h_in = Input(shape=(cell_size, ), name='h_in')
        state_c_in = Input(shape=(cell_size, ), name='c_in')
        seq_in = Input(shape=(), name='seq_in', dtype=tf.int32)

        # hidden_out = make_mlp(obs_in, layer_size, num_layers=num_layers - 1, 
        #         layer_size=layer_size, activation=activation, 
        #         name_prefix='policy_hidden_fc_block1_')
        hidden_out = Dense(layer_size, activation=activation)(obs_in)

        print('cell_size:', cell_size)
        lstm_out, state_h, state_c = LSTM(cell_size, return_sequences=True,
            return_state=True, name='lstm')(
                inputs=hidden_out,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_h_in, state_c_in])
        
        flat_action_params = Dense(
            self.num_outputs, 
            activation=tf.keras.activations.linear, 
            name='policy_flat_action_prams')(lstm_out)
        
        value_func_out = Dense(1, activation=None, name='vf_out')(lstm_out)
        # value_func_out = make_mlp(obs_in, 1, num_layers=num_layers,
        #         layer_size=layer_size, activation=activation, name_prefix='vf_hidden_',
        #         output_name='vf_out')

        self.base_model = Model(inputs=[obs_in, seq_in, state_h_in, state_c_in],
                outputs=[flat_action_params, value_func_out, state_h, state_c])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # flat_action_params = make_mlp(obs_in, num_outputs, num_layers=num_layers, 
        #         layer_size=layer_size, activation=activation, name_prefix='policy_hidden_',
        #         output_name='policy_flat_action_prams')
        # value_func_out = make_mlp(obs_in, 1, num_layers=num_layers, 
        #         layer_size=layer_size, activation=activation, name_prefix='vf_hidden_',
        #         output_name='vf_out')
        # self.base_model = Model(obs_in, [flat_action_params, value_func_out])
        # self.register_variables(self.base_model.variables)
        # self.base_model.summary()
    
    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        print('forward_rnn$seq_lens:', seq_lens)
        model_out, self._value_out, h, c = self.base_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
    
    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("rnn_model", RNNModel)