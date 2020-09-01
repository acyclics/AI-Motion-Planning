import os
import sys
import numpy as np

from environment.module.module import EnvModule
from mujoco_worldgen.util.types import store_args

class LidarSites(EnvModule):
    '''
    Adds sites to visualize Lidar rays
        Args:
            n_agents (int): number of agents
            n_lidar_per_agent (int): number of lidar sites per agent
    '''
    @store_args
    def __init__(self, n_agents, n_lidar_per_agent):
        pass

    def build_world_step(self, env, floor, floor_size):
        for i in range(self.n_agents):
            for j in range(self.n_lidar_per_agent):
                floor.mark(f"agent{i}:lidar{j}", (0.0, 0.0, 0.0), rgba=np.zeros((4,)))
        return True

    def modify_sim_step(self, env, sim):
        # set lidar size and shape
        self.lidar_ids = np.array([[sim.model.site_name2id(f"agent{i}:lidar{j}")
                                    for j in range(self.n_lidar_per_agent)]
                                   for i in range(self.n_agents)])
        # set lidar site shape to cylinder
        sim.model.site_type[self.lidar_ids] = 5
        sim.model.site_size[self.lidar_ids, 0] = 0.02
