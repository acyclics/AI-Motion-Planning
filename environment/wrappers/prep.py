import sys
import numpy as np
from copy import deepcopy
from itertools import compress

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple

from mujoco_worldgen.util.rotation import mat2quat
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix
from environment.wrappers.util_w import update_obs_space
from environment.utils.vision import insight, in_cone2d


class PrepWrapper(gym.Wrapper):
    '''
        Add variables and mechanisms needed before any wrapper.
    '''
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Store agents' qpos
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        return self.observation(obs)

    def observation(self, obs):
        return obs        

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        sim = self.unwrapped.sim
        # Check if agent spawned out of arena
        for aqidx in self.agent_qpos_idxs:
            agent_qpos = sim.data.qpos[aqidx]
            if agent_qpos[0] < 0 or agent_qpos[0] > 8540.50 or agent_qpos[1] < 0 or agent_qpos[1] > 5540.50:
                done = True
        rew = 0.0
        return self.observation(obs), rew, done, info
