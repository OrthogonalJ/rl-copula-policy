import numpy as np
import gym
from gym import Env
from gym.spaces import Dict, Box, Tuple
from action_dist_prac.utils.gym_space_utils import convert_to_flat_tuple_space

class LatentVariableGymEnvWrapper(Env):
    def __init__(self, config):
        self._base_env = gym.make(config['env_name'])
        self.observation_space = self._base_env.observation_space
        self._base_action_spaces = convert_to_flat_tuple_space(self._base_env.action_space)
        # self._base_action_spaces = [self._base_env.action_space] \
        #         if not isinstance(self._base_env.action_space, Tuple) \
        #         else list(self._base_env.action_space)
                #else [space for space in self._base_env.action_space]
        #self._base_action_spaces = list(reversed(self._base_action_spaces))

        self._latent_space = Box(0.0, 1.0, shape=(len(self._base_action_spaces),))

        #self._latent_space = Box(np.finfo(np.float32).min, np.finfo(np.float32).max, 
        #        shape=(len(self._base_action_spaces),))
        #self._latent_cdf_vals_space = Box(0.0, 1.0, shape=(len(self._base_action_spaces),))

        self.action_space = Tuple([self._latent_space] + self._base_action_spaces)
    
    def reset(self):
        return self._base_env.reset()
    
    def step(self, action):
        print('ENV action:', action)
        if len(self._base_action_spaces) == 1:
            base_action = action[1]
        else:
            base_action = list(action[1:])
            #base_action = list(reversed(action[1:]))
        obs, reward, done, info = self._base_env.step(base_action)
        info['latent_variable'] = action[0].tolist()
        return obs, reward, done, info