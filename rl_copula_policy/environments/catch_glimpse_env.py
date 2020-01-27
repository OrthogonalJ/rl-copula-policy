import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from ray.tune import register_env # pylint: disable=import-error

from rl_vis_attention.catch_game.catch_flat_env import CatchFlatEnv # pylint: disable=import-error

class CatchGlimpseEnv(gym.Env):
    """
    Observation Space: Tuple of
        image: Box((w*h,))
        next_location: Box((2))
    
    Action Space: Tuple of
        action: Discrete(3)
        location: Box((2,))
    """
    def __init__(self, env_config):
        self._env = CatchFlatEnv(**env_config)
        location_space = Box(np.finfo(np.float32).min , np.finfo(np.float32).max, (2,), np.float32)
        self.action_space = Tuple([Discrete(3), location_space])
        self.observation_space = Tuple([
            Box(0, 255, (self._env._field_size * self._env._field_size, ), np.int32),
            location_space
        ])
        self._next_location = None
        self.reset()

    def reset(self):
        self._next_location = np.zeros((2,), np.float32)
        image = self._env.reset()
        return (image, self._next_location)

    def step(self, 
            action_loc_tuple=None, action=None,
            location=None, last_location=None # added for interface compatibility with ray_prac.glimpse_net_evaluation.sample_trajectory
        ):
        if location is None:
            action, self._next_location = action_loc_tuple
        else:
            action = action_loc_tuple
            self._next_location = location
        image, reward, done, info = self._env.step(action)
        obs = (image, self._next_location)
        return obs, reward, done, info


def make_env(config):
    return CatchGlimpseEnv(config)
register_env('catch_glimpse', make_env)
