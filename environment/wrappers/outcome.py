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


class OutcomeWrapper(gym.Wrapper):
    '''
        Adds reward according to outcome of match.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = 4
        self.observation_space = update_obs_space(self.env, {'destinations': [2, 2]})

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get new position
        self.destinations = np.array([[np.random.uniform(0.0, 8080), np.random.uniform(0.0, 4480)],
                                      [np.random.uniform(0.0, 8080), np.random.uniform(0.0, 4480)]])
        # Get agent xy qpos
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        return self.observation(obs)

    def observation(self, obs):
        temp_dests = deepcopy(self.destinations)
        temp_dests[:, 0] = self.destinations[:, 0] / 8080
        temp_dests[:, 1] = self.destinations[:, 1] / 4480
        obs['destinations'] = temp_dests
        return obs
    
    def visualize_markers(self):
        sim = self.unwrapped.sim
        marker1_idx = sim.model.site_name2id("dest_marker1")
        marker2_idx = sim.model.site_name2id("dest_marker2")
        sim.data.site_xpos[marker1_idx][0:2] = self.destinations[0]
        sim.data.site_xpos[marker2_idx][0:2] = self.destinations[1]

    def get_distance_reward(self):
        agent1_pos = self.unwrapped.sim.data.qpos[self.agent_qpos_idxs[0]][0:2]
        agent2_pos = self.unwrapped.sim.data.qpos[self.agent_qpos_idxs[1]][0:2]
        agent1_rew = -np.linalg.norm(self.destinations[0] - agent1_pos) / np.sqrt(8080**2 + 4480**2)
        agent2_rew = -np.linalg.norm(self.destinations[1] - agent2_pos) / np.sqrt(8080**2 + 4480**2)
        dist_rew = agent1_rew + agent2_rew
        return dist_rew

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visualize_markers()
        dist_rew = self.get_distance_reward()
        rew += dist_rew
        return self.observation(obs), rew, done, info
