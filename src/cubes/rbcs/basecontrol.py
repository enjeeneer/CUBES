"""Base class for controllers"""

from abc import ABC, abstractmethod


class BaseControl(ABC):
    """Base class for controllers"""

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs_dict, action_dict, action_range_dict):
        ...
