# pylint: disable=all

"""
CUBES' multi-task and meta RL benchmarks.

Module designed to closely resemble Bauwerk (https://github.com/rdnfn/bauwerk/blob/main/bauwerk/benchmarks.py)
by Arduin Findeis, which itself is similar to Meta-World (https://github.com/rlworkgroup/metaworld).
"""

import cubes
import abc
import gym
import dataclasses
import numpy as np

from dataclasses import dataclass
from typing import List, Any, Optional, Dict, Union

ENV_NAME = "test"


@dataclass
class Task:
    """
    All data required to define the MDP of one CUBES env.
    Should be passed to the set_task method.
    """

    cfg: object
    env_name: str


@dataclass
class ParamDist:
    fn: Any  # function to draw params from


@dataclass()
class ContParamDist(ParamDist):
    """
    Distribution over single cfg param.
    """

    low: float  # lower bound of dist
    high: float  # upper bound of dist

    def sample(self):
        return self.fn(low=self.low, high=self.high)


def sample_cfg_dist(self) -> cubes.EnvConfig:
    """Sample from CfgDist."""

    params = dict(
        (field.name, getattr(self, field.name).sample())
        if isinstance(getattr(self, field.name), ParamDist)
        else (field.name, getattr(self, field.name))
        for field in dataclasses.fields(self)
    )
    return cubes.EnvConfig(**params)


CfgDist = dataclasses.make_dataclass(
    cls_name="CfgDist",
    fields=list(
        (field.name, Union[field.type, ParamDist], field)
        for field in dataclasses.fields(cubes.EnvConfig)
    ),
    namespace={
        "sample": sample_cfg_dist,
    },
)


class Benchmark(abc.ABC):
    """
    Abstract class for CUBES' benchmarks
    """

    @abc.abstractmethod
    def __init__(self, seed=None):
        pass

    @property
    def train_classes(self) -> Dict:
        """Gets all environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> Dict:
        """Gets all environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> Dict:
        """Gets all training tasks used for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> Dict:
        """Gets all test tasks used for this benchmark."""
        return self._test_tasks

    @property
    def make_env(self) -> gym.Env:
        """Create env instance one which we set tasks."""


class AbstractBuildDist(Benchmark):
    def __init__(
        self,
        cfg_dist: CfgDist,
        num_train_tasks: int,
        num_test_tasks: int,
        dtype: Union[str, np.dtype] = None,
        episode_len: Optional[int] = None,
        seed: Optional[int] = None,
        env_kwargs: Optional[Dict] = None,
    ):
        """
        Abstract building distribution
        """
        super().__init__()

        self.cfg_dist = cfg_dist
        self.num_train_tasks = num_train_tasks
        self.num_test_tasks = num_test_tasks
        self.dtype = dtype
        self.episode_len = episode_len
        self.seed = seed
        self.env_kwargs = env_kwargs

    def _create_tasks(self, seed, num_tasks):
        """Creates tasks for the distribtuion of buildings"""
        if seed is not None:
            old_np_state = np.random.get_state()
            np.random.seed(seed)

        tasks = []

        for _ in range(num_tasks):
            task = Task(env_name=ENV_NAME, cfg=self.cfg_dist.sample())
            tasks.append(task)

        if seed is not None:
            np.random.set_state(old_np_state)

        return tasks

    def make_env(self):
        """Creates environment for a given cfg dist."""
        cfg = self.cfg_dist.get_default_env_cfg()
        for name, value in self.env_kwargs.items():
            setattr(cfg, name, value)

        env = gym.make(ENV_NAME, cfg=cfg)

        return env


### SPECIFIC BUILD DISTS TBC ###
