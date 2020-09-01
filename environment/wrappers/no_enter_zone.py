import sys
import numpy as np
from copy import deepcopy
from itertools import compress

import gym

from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix


class NoEnterZoneWrapper(gym.Wrapper):
    '''
        Penalizes an agent for entering specific zones.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = 4

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get agent qpos idx
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        return self.observation(obs)

    def observation(self, obs):
        return obs
    
    def agent_zone(self, sim):
        rew = np.array([0.0 for _ in range(2)])
        for idx in range(2):
            is_zone_entered = sim.data.get_sensor(f'agent{idx}:no_enter_zone_touch')
            if is_zone_entered > 0:
                rew[idx] += -0.1
        return rew

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        sim = self.unwrapped.sim
        rew += self.agent_zone(sim)
        return self.observation(obs), rew, done, info
