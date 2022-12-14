"""
Define custom reward functions
"""
from sinergym.utils.rewards import BaseReward
from gym import Env
from typing import Any, Dict, Tuple


class HCLoadsReward(BaseReward):
    """
    make a simple reward function based on sum of heating and cooling loads
    """

    def __init__(self, env: Env, heating_variable: str, cooling_variable: str):
        super().__init__(env)

        # Name of the variables
        self.heating_name = heating_variable
        self.cooling_name = cooling_variable

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary
            with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        reward_heating = -obs_dict[self.heating_name]
        reward_cooling = -obs_dict[self.cooling_name]

        # Weighted sum of both terms
        reward = reward_heating + reward_cooling

        reward_terms = {
            "reward_heating": reward_heating,
            "reward_cooling": reward_cooling,
        }

        return reward, reward_terms
