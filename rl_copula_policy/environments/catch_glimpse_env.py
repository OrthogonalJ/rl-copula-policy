import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from ray.tune import register_env  # pylint: disable=import-error
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator

from rl_vis_attention.catch_game.catch_flat_env import CatchFlatEnv  # pylint: disable=import-error


class CatchGlimpseEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
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
        location_space = Box(
            -1.0,
            # np.finfo(np.float32).min, 
            1.0,
            # np.finfo(np.float32).max,
            (2,), 
            np.float32
        )
        self.action_space = Tuple([Discrete(3), location_space])
        self.observation_space = Tuple([
            Box(0, 255, (self._env._field_size * self._env._field_size, ),
                np.int32), location_space
        ])
        self._next_location = None

        self._fig = None
        self._timestep = None
        self._last_reward = None
        self._last_location = None

        self.reset()

    def reset(self):
        self._fig = None
        self._timestep = 0
        self._last_reward = 0
        self._last_location = np.zeros((2, ), np.float32)

        self._next_location = np.zeros((2, ), np.float32)
        image = self._env.reset()
        return (image, self._next_location)

    def step(
            self,
            action_loc_tuple=None,
            action=None,
            location=None,
            last_location=None  # added for interface compatibility with ray_prac.glimpse_net_evaluation.sample_trajectory
    ):
        # book keeping for render method
        self._timestep += 1
        self._last_location = self._next_location

        if location is None:
            action, self._next_location = action_loc_tuple
        else:
            action = action_loc_tuple
            self._next_location = location

        image, self._last_reward, done, info = self._env.step(action)
        obs = (image, self._next_location)
        return obs, self._last_reward, done, info

    def render(self, mode='rgb_array', info=None):
        if mode != 'rgb_array':
            return super(CatchGlimpseEnv, self).render(mode)

        if info is None:
            return

        if self._fig is None:
            self._fig = plt.figure()

        # Glimpse dimensions
        base_image_height = self._env._field_size
        base_image_width = self._env._field_size

        # Display settings
        pixel_min = 0
        pixel_max = 255
        extent = (-1, 1, -1, 1)
        colour_map = 'PuBuGn'
        show_image = lambda ax, img: ax.imshow(
            img, cmap=colour_map, vmin=0.0, vmax=1.0, extent=extent)

        n_patches = info['n_patches']
        glimpse = info['glimpse']
        initial_glimpse_size = info['initial_glimpse_size']

        glimpse_patches = [
            np.squeeze(glimpse[..., p_idx]) for p_idx in range(n_patches)
        ]

        location = self._last_location

        self._fig.clf()
        outer_grid_spec = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.2, 
                hspace=0.2)
        observation_ax = plt.Subplot(self._fig, outer_grid_spec[0])
        patch_grid = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=n_patches,
                subplot_spec=outer_grid_spec[1], wspace=0.1, hspace=0.1)

        # Draw glimpse patches in right half of figure
        for p_idx in range(n_patches):
            patch_ax = plt.Subplot(self._fig, patch_grid[p_idx])
            show_image(patch_ax, glimpse_patches[p_idx].astype(np.float32))
            if p_idx == 0:
                patch_ax.set_title('Glimpse (Location: {})'.format(
                    np.round(location, 2)))
            self._fig.add_subplot(patch_ax)

    
        # Draw observation image in left half of figure
        observation = np.squeeze(self._env._field)
        observation = (observation - pixel_min) / (pixel_max - pixel_min)
        show_image(observation_ax, observation)
        observation_ax.set_title('Observation (Timestep: {}, Pixel Min: {}, Pixel Max: {}, Return: {})' \
                .format(self._timestep, observation.min(), observation.max(), self._last_reward))

        # Overlay grid on base image
        grid_unit_y = 1 / (self._env._field_size / 2)
        grid_unit_x = 1 / (self._env._field_size / 2)
        y_tick_loc = MultipleLocator(base=grid_unit_y)
        x_tick_loc = MultipleLocator(base=grid_unit_x)
        observation_ax.xaxis.set_major_locator(x_tick_loc)
        observation_ax.yaxis.set_major_locator(y_tick_loc)
        observation_ax.grid(which='major', axis='both', linestyle='-')

        ## Draw rectangle to show location of glimpse in observation ##

        for p_idx in range(n_patches):
            patch_size = initial_glimpse_size * (2**p_idx)
            normed_patch_width = (patch_size / base_image_width) * 2
            normed_patch_height = (patch_size / base_image_height) * 2
            
            # Bottom left corner coords
            patch_corner_y = (-1 * location[0]) - normed_patch_height/2
            patch_corner_x = location[1] - normed_patch_width/2

            # snap to pixel grid (not sure if this is correct)
            # grid_unit_y = 2 / (base_image_height)
            # grid_unit_x = 2 / (base_image_width)
            # patch_corner_y = np.ceil(np.abs(patch_corner_y / grid_unit_y)) * (grid_unit_y) * np.sign(patch_corner_y)
            # patch_corner_x = np.ceil(np.abs(patch_corner_x / grid_unit_x)) * (grid_unit_x) * np.sign(patch_corner_x)

            rect = Rectangle((patch_corner_x, patch_corner_y),
                             normed_patch_width,
                             normed_patch_height,
                             fill=False,
                             edgecolor='black')
            observation_ax.add_artist(rect)

        self._fig.add_subplot(observation_ax)

        plt.draw()
        return np.array(self._fig.canvas.renderer.buffer_rgba())

def make_env(config):
    return CatchGlimpseEnv(config)


register_env('catch_glimpse', make_env)
