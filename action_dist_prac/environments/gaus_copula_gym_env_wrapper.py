import numpy as np
import gym
from gym import Env
from gym.spaces import Dict, Box, Tuple
from action_dist_prac.utils.gym_space_utils import convert_to_flat_tuple_space

class GausCopulaGymEnvWrapper(Env):
    def __init__(self, config):
        self._base_env = gym.make(config['env_name'])
        self.observation_space = self._base_env.observation_space
        self._base_action_spaces = convert_to_flat_tuple_space(self._base_env.action_space)
        
        #self._latent_space = Box(0.0, 1.0, shape=(len(self._base_action_spaces),))

        self._latent_space = Box(np.finfo(np.float32).min, np.finfo(np.float32).max, 
                shape=(len(self._base_action_spaces),))
        self._latent_cdf_vals_space = Box(0.0, 1.0, shape=(len(self._base_action_spaces),))

        self.action_space = Tuple([self._latent_cdf_vals_space, self._latent_space] + self._base_action_spaces)
    
    def reset(self):
        return self._base_env.reset()
    
    def step(self, action):
        # print('ENV action:', action)
        copula_cdf_vals = action[0]
        copula_latent_variable = action[1]
        if len(self._base_action_spaces) == 1:
            base_action = action[2]
        else:
            base_action = list(action[2:])
        obs, reward, done, info = self._base_env.step(base_action)
        # Calling tolist so that values are JSON serializable (required for rllib env traces)
        info['latent_variable'] = copula_latent_variable.tolist()
        info['copula_cdf_vals'] = copula_cdf_vals.tolist()
        return obs, reward, done, info
