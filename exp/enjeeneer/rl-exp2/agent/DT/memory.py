import torch
import numpy as np
from typing import Dict



class Memory:
    def __init__(self, cfg, trajectories):
        self.cfg = cfg
        if len(trajectories) <= cfg.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj['rewards'].sum() for traj in trajectories]
            sort_idxs = np.argsort(returns)  # ascending order
            self.trajectories = [trajectories[i] for i in sort_idxs[-cfg.capacity:]]

        self.idx = 0

    def store(self, traj: Dict):
        """
        Stores trajectory in memory.
        :param traj:
        :return:
        """
        if len(self.trajectories) < self.cfg.capacity:
            self.trajectories.extend(traj)
        else:
            # first-in-first-out
            self.trajectories[1:] = self.trajectories[:-1]
            self.trajectories[0] = traj

    def sample(self):
        pass


