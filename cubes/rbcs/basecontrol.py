"""Base class for controllers"""

from typing import Dict, List
from abc import ABC, abstractmethod


class BaseControl(ABC):
    """Base class for controllers"""

    def __init__(self):
        pass

    @abstractmethod
    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        action_range_dict: Dict[str, List] = None,
    ) -> Dict[str, float]:
        pass
