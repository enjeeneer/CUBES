"""Utilites using gym package separate from other utilites,
so that gym does not have to be loaded unnecessarily"""

from gym.spaces import Box
import numpy as np


def get_space(var_list, is_observation_space):
    if is_observation_space:
        lower_limits = np.zeros(len(var_list) + 4)  # sinergym adds time info
        upper_limits = np.zeros(len(var_list) + 4)

        lower_limits[0:4] = [0, 0, 0, 0]
        upper_limits[0:4] = [3000, 12, 31, 24]

        for iv, v in enumerate(var_list):
            lower_limits[iv + 4], upper_limits[iv + 4] = v.get_range()

    else:
        lower_limits = np.zeros(len(var_list))
        upper_limits = np.zeros(len(var_list))

        for iv, v in enumerate(var_list):
            lower_limits[iv], upper_limits[iv] = v.get_range()

    return Box(
        low=lower_limits,
        high=upper_limits,
        dtype=np.float32,
    )
