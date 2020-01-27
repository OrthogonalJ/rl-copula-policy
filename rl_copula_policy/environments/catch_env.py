import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from ray.tune import register_env # pylint: disable=import-error

from rl_vis_attention.catch_game.catch_flat_env import CatchFlatEnv # pylint: disable=import-error

class CatchEnv(gym.Env):
    def __init__(self, env_config):
        self._env = CatchFlatEnv(**env_config)
        self.action_space = Discrete(3)
        self.observation_space = Box(0, 255, (self._env._field_size * self._env._field_size, ), np.int32)
        self.reset()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

def make_env(config):
    return CatchEnv(config)
register_env('catch', make_env)
