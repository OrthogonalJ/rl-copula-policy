from enum import Enum
import numpy as np
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models import ModelCatalog
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Reshape, Flatten, TimeDistributed, Activation, Concatenate
import tensorflow_probability as tfp

from rl_copula_policy.utils.utils import shape_list

class BaselineInputTypes:
    image = 'image'
    core_net = 'core_net'


#### HELPERS ####

def collapse_time_into_batch(tensor, use_keras=False):
    """
    Args:
        tensor(tensor): A tensor flow tensor with shape (batches, time steps, <data dims>...)
    
    Returns(tensor): A reshaped version of tensor with timesteps pivoted into the batch dimension;
            with one batch for each timestep in each batch in the original tensor.
    """
    def _reshape_impl(tensor, include_batch_dim=True):
        shape = tf.shape(tensor)
        new_shape = tf.concat([[-1], shape[2:]], axis=0) if include_batch_dim \
                else shape[2:]
        tensor_reshaped = tf.reshape(tensor, new_shape)
        return tensor_reshaped
    
    if use_keras:
        args = {'include_batch_dim': False}
        reshape_layer = keras.layers.Lambda(_reshape_impl, arguments=args)
        tensor_reshaped = reshape_layer(tensor)
    else:
        tensor_reshaped = _reshape_impl(tensor)
    
    return tensor_reshaped

def uncollapse_time_from_batch(tensor, seq_len):
    """
    Args:
        tensor(tensor): A tensorflow tensor with shape (batch, <data dims>)
        seq_len(int): Sequence length (all sequences are assumed to have this length)
    Returns: Reshaped tensor with shape (batches, seq_len, <data dims>...). 
            Assumes that every seq_len batches are independent sequences.
    """
    # print('uncollapse_time_from_batch: (tensor, seq_len):', (tensor, seq_len))
    shape = tf.shape(tensor)
    tensor_batch_time = tf.reshape(tensor, tf.concat([[-1], [seq_len], shape[1:]], axis=0))
    return tensor_batch_time

#### MODEL ####

class GlimpseNetModel(RecurrentTFModelV2):
    LOCATION_DIMS = 2
    DEFAULT_BASELINE_INPUT_TYPE = 'image'
    """
    Model Config:
        obs_shape
        n_patches
        initial_glimpse_size
        action_dim
        location_std
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config,
            name):

        super(GlimpseNetModel, self).__init__(obs_space, action_space,
                num_outputs, model_config, name)

        self.core_net_cell_size = 256

        custom_options = model_config['custom_options']
        self.observation_shape = custom_options['obs_shape']
        self.n_patches = custom_options['n_patches']
        self.initial_glimpse_size = custom_options['initial_glimpse_size']
        self.glimpse_net_activation = custom_options.get('glimpse_net_activation', 'relu')
        self.action_dim = custom_options['action_dim']
        self.location_std_value = custom_options.get('location_std', None)
        self.sep_location_net_gradients = custom_options['sep_location_net_gradients']
        self.baseline_input_type = custom_options.get('baseline_input_type', 
                GlimpseNetModel.DEFAULT_BASELINE_INPUT_TYPE)
        
        flat_image_len = np.product(self.observation_shape)
        self.image_flat_in = Input(shape=(None, flat_image_len), name='image_flat_in')
        self.last_location_in = Input(shape=(None, 2, ), name='last_location_in')
        self.state_in_h = Input(shape=(self.core_net_cell_size, ), name='h_in')
        self.state_in_c = Input(shape=(self.core_net_cell_size, ), name='c_in')
        self.seq_lens_in = Input(shape=(), name='seq_in', dtype=tf.int32)
        # print('seq_lens_in:', self.seq_lens_in)
        # All sequences are assumed to be padded to this length
        max_seq_len = tf.reduce_max(self.seq_lens_in)

        # Removing time dim to match input shape provide in TFModel v1
        self.image_flat_bf = collapse_time_into_batch(self.image_flat_in)
        self.last_location_bf = collapse_time_into_batch(self.last_location_in)
        # print('last_location_bf:', last_location_bf)

        image_bf = Reshape(self.observation_shape)(self.image_flat_bf)

        glimpse_out_bf = self.build_glimpse_network(image_bf, self.last_location_bf, 
                self.initial_glimpse_size)
        
        # print('glimpse_out_bf', glimpse_out_bf)
        glimpse_out = uncollapse_time_from_batch(glimpse_out_bf, max_seq_len)

        rnn_out, state_h, state_c = self.build_core_net(glimpse_out, self.seq_lens_in, 
                self.state_in_h, self.state_in_c)

        self.action_params = self.build_action_net(rnn_out, self.action_dim)

        self.loc_means, self.loc_logstds = self.build_location_net(rnn_out)
        location_params = tf.concat([self.loc_means, self.loc_logstds], axis=-1)

        output_tensor = tf.concat([self.action_params, location_params], axis=-1)

        value_function_out = self._build_value_function(self.baseline_input_type, 
                core_model_out=rnn_out, image_flat=self.image_flat_in)
        
        self.base_model = keras.Model(inputs=self.model_inputs(), 
                outputs=[output_tensor, value_function_out, state_h, state_c])
        self.register_variables(self.base_model.variables)
        # self.base_model.summary()

        self.glimpse_model = keras.Model(inputs=self.model_inputs(), 
                outputs=[self.glimpse])

    def model_inputs(self):
        return [self.image_flat_in, self.last_location_in, self.seq_lens_in, 
                self.state_in_h, self.state_in_c]

    def forward_rnn(self, inputs, state, seq_lens):
        inputs = self._parse_observation(inputs)
        model_inputs = [inputs['image'], inputs['last_location'], seq_lens] + state
        model_out, self._value_out, h, c = self.base_model(model_inputs)
        self._glimpse_out = self.glimpse_model(model_inputs)
        return model_out, [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.core_net_cell_size, np.float32),
            np.zeros(self.core_net_cell_size, np.float32),
        ]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def extra_compute_action_fetches(self):
        return {
            'glimpse': self._glimpse_out, 
            'n_patches': tf.constant([self.n_patches], dtype=tf.int32),
            'initial_glimpse_size': tf.constant([self.initial_glimpse_size], dtype=tf.int32)
        }
    
    #### PRIVATE METHODS ####

    def _parse_observation(self, obs):
        return {
            'image': obs[..., :-2],
            'last_location': obs[..., -2:]
        }

    def _build_value_function(self, baseline_input_type, core_model_out, image_flat):
        input_tensor = self._select_value_function_input(baseline_input_type, 
                core_model_out, image_flat)
        output_layer = TimeDistributed(Dense(64, activation=self.glimpse_net_activation))(input_tensor)
        output_layer = TimeDistributed(Dense(64, activation=self.glimpse_net_activation))(output_layer)
        output_layer = TimeDistributed(Dense(64, activation=self.glimpse_net_activation))(output_layer) 
        output_layer = TimeDistributed(Dense(1, activation=None))(output_layer)
        # output_layer = TimeDistributed(Reshape([-1]))(output_layer)
        return output_layer
    
    def _select_value_function_input(self, baseline_input_type, core_model_out, image_flat):
        input_types = {
            BaselineInputTypes.image: image_flat, 
            BaselineInputTypes.core_net: core_model_out
        }
        return input_types[baseline_input_type]

    def build_action_net(self, input_tensor, action_dim):
        output_layer = Dense(action_dim, activation=None)
        output_tensor = TimeDistributed(output_layer)(input_tensor)
        return output_tensor

    def build_location_net(self, input_tensor):
        output_layer = Dense(2, activation=None)
        means = TimeDistributed(output_layer)(input_tensor)

        class LocationDistParametersLayer(keras.layers.Layer):
            def __init__(self, location_std_value, **kwargs):
                super(LocationDistParametersLayer, self).__init__(**kwargs)
                self._location_std_value = location_std_value
            
            def build(self, input_shape):
                if self._location_std_value is None:
                    # learn std
                    self._logstds = self.add_weight(
                        name='logstds',
                        shape=(2,),
                        initializer='uniform',
                        trainable=True
                    )
                else:
                    # use fixed std
                    stds = keras.backend.constant(
                        [self._location_std_value, self._location_std_value], 
                        dtype=tf.float32
                    )
                    self._logstds = keras.backend.log(stds)
                super(LocationDistParametersLayer, self).build(input_shape)

            def call(self, means):
                num_non_data_dims = len(shape_list(means)) - 1
                single_instance_shape = ([1] * num_non_data_dims) + [2]
                logstd_output = tf.reshape(self._logstds, single_instance_shape)
                logstd_output = tf.tile(
                    logstd_output, 
                    tf.concat([tf.shape(means)[:-1], [1]], axis=0)
                )
                return [means, logstd_output]
            
            def compute_output_shape(self, input_shape):
                return [input_shape, input_shape[:-1] + [2]]

        params_layer = LocationDistParametersLayer(self.location_std_value)
        means, logstds = params_layer(means)
        return means, logstds
    
    def build_core_net(self, glimpse_out, seq_len, state_in_h, state_in_c):
        initial_state = [state_in_h, state_in_c]
        # print('[build_core_net] glimpse_out:', glimpse_out)
        # print('[build_core_net] tf.sequence_mask(seq_len):', tf.sequence_mask(seq_len))
        # print('[build_core_net] initial_state:', initial_state)

        lstm_layer = LSTM(self.core_net_cell_size, return_sequences=True, return_state=True,
                name='core_net_lstm')
        
        rnn_output, state_h, state_c = lstm_layer(
                inputs=glimpse_out, 
                mask=tf.sequence_mask(seq_len), 
                initial_state=initial_state)
        
        return rnn_output, state_h, state_c

    def build_glimpse_sensor(self, image, location, initial_glimpse_size):
        """
        Args:
            image(tensor)
            location(tensor)
            initial_glimpse_size(tensor)
        """
        # TODO: Figure out if initial_glimpse_size arg is needed (if not just use self.initial_glimpse_size)

        def _build_glimpse_sensor_impl(input_tensors,  # (image, location) tensors
                initial_glimpse_size, n_patches, observation_shape # config
                ):
            image, location = input_tensors
            # print('location:', location)
            # print('image:', image)

            # location = tf.clip_by_value(location, -1.0, 1.0)

            initial_glimpse_size_tensor = tf.constant([initial_glimpse_size, initial_glimpse_size], 
                    dtype=tf.int32)
            
            # Pad image with zeros to give noiseless pixels when glimpses 
            # overlap with the image's true boundary
            # max_patch_radius = int(initial_glimpse_size * (2 ** (n_patches - 1)) / 2)
            # image = tf.pad(image, [[0, 0], [max_patch_radius, max_patch_radius], 
            #         [max_patch_radius, max_patch_radius], [0, 0]])
            # print('image after padding:', image)

            # base_image_height = observation_shape[0]
            # base_image_width = observation_shape[1]
            # location_y = location[..., 0] * base_image_height/2 / (base_image_height/2 + max_patch_radius)
            # location_x = location[..., 1] * base_image_width/2 / (base_image_width/2 + max_patch_radius)
            # location = tf.stack([location_y, location_x], axis=-1)
            # # must match the batch shape of image
            # location = tf.reshape(location, shape_list(image)[:-3] + [2])
            # # print('location after clipping and scaling:', location)

            patches = []
            for i in range(n_patches):
                patch_size = initial_glimpse_size_tensor * (2 ** i)
                patch = tf.image.extract_glimpse(image, size=patch_size, offsets=location,
                        normalized=True, centered=True, noise='zero')
                # print('patch {} before resize: {}'.format(i, patch))
                
                # RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
                RESIZE_METHOD = tf.image.ResizeMethod.GAUSSIAN

                patch = tf.image.resize(patch, size=initial_glimpse_size_tensor,
                        method=RESIZE_METHOD)
                
                # print('patch {} after resize: {}'.format(i, patch))
                patches.append(patch)
            # print('[_build_glimpse_sensor_impl] patches:', [p for p in patches])
            glimpse = tf.stack(patches, axis=-1)
            return glimpse

        
        config = {'initial_glimpse_size': initial_glimpse_size, 
                'n_patches': self.n_patches, 'observation_shape': self.observation_shape}
        return _build_glimpse_sensor_impl((image, location), **config)
        
    
    def build_glimpse_network(self, image, location, initial_glimpse_size):
        """
        Args:
            image(tensor)
            location(tensor)
            initial_glimpse_size(tensor)
        """
        self.glimpse = self.build_glimpse_sensor(image, location, initial_glimpse_size)
        # print('self.glimpse:', self.glimpse)
        glimpse_flat = Flatten()(self.glimpse)
        # print('glimpse_flat:', glimpse_flat)
        glimpse_hidden_layer_output = Dense(
                units=128, activation=self.glimpse_net_activation)(glimpse_flat)
        # print('glimpse_hidden_layer_output:', glimpse_hidden_layer_output)

        location_hidden_layer_output = Dense(
                units=128, activation=self.glimpse_net_activation)(location)
        # print('location_hidden_layer_output:', location_hidden_layer_output)

        # Option 1: Relu(Linear(h_g) + Linear(h_l)), h_g = Relu(Linear(glimpse)), h_l = Relu(Linear(loc))
        combined_layer_output = Dense(256, activation=None)(glimpse_hidden_layer_output) \
                + Dense(256, activation=None)(location_hidden_layer_output)
        # combined_layer_output = Concatenate(axis=-1)([
        #     Dense(128, activation=None)(glimpse_hidden_layer_output),
        #     Dense(128, activation=None)(location_hidden_layer_output)
        # ])
        combined_layer_output = Activation(self.glimpse_net_activation)(combined_layer_output)

        # Option 2:
        #combined_layer_output = glimpse_hidden_layer_output + location_hidden_layer_output
        #combined_layer_output = Dense(units=128, activation=self.glimpse_net_activation)(combined_layer_output)

        return combined_layer_output

ModelCatalog.register_custom_model("glimpse_net_model", GlimpseNetModel)
