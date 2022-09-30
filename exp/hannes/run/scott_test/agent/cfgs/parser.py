from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig


def parse_cfg() -> Union[DictConfig, ListConfig]:
    """
    Parses agent and env configs files, adds c02 data and returns OmegaConf object
    """
    agent_cfg_path = "agent/cfgs/sac.yaml"
    base = OmegaConf.load(agent_cfg_path)

    return base
