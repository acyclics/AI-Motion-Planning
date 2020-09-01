import numpy as np
import time

import glfw
from mujoco_py import const, MjViewer
from gym.spaces import Box, MultiDiscrete, Discrete


class EnvHandler():

    def __init__(self, env):
        self.env = env
        self.action_types = list(self.env.action_space.spaces.keys())
        self.num_action_types = len(self.env.action_space.spaces)
        self.num_action = self.num_actions(self.env.action_space)
        self.agent_mod_index = 0
        self.action_mod_index = 0
        self.action_type_mod_index = 0
        self.action = self.zero_action(self.env.action_space)

    def num_actions(self, ac_space):
        n_actions = []
        for k, tuple_space in ac_space.spaces.items():
            s = tuple_space.spaces[0]
            if isinstance(s, Box):
                n_actions.append(s.shape[0])
            elif isinstance(s, Discrete):
                n_actions.append(1)
            elif isinstance(s, MultiDiscrete):
                n_actions.append(s.nvec.shape[0])
            else:
                raise NotImplementedError(f"not NotImplementedError")

        return n_actions

    def zero_action(self, ac_space):
        ac = {}
        for k, space in ac_space.spaces.items():
            if isinstance(space.spaces[0], Box):
                ac[k] = np.zeros_like(space.sample())
            elif isinstance(space.spaces[0], Discrete):
                ac[k] = np.ones_like(space.sample()) * (space.spaces[0].n // 2)
            elif isinstance(space.spaces[0], MultiDiscrete):
                ac[k] = np.ones_like(space.sample(), dtype=int) * (space.spaces[0].nvec // 2)
            else:
                raise NotImplementedError("MultiDiscrete not NotImplementedError")
                # return action_space.nvec // 2  # assume middle element is "no action" action
        return ac

    def observation(self, obs):
        obs_self = obs['observation_self']
        dests = obs['destinations']
        obs = np.concatenate([obs_self[0], obs_self[1], obs_self[2, 0:3], obs_self[3, 0:3], dests[0], dests[1]])
        return obs

    def reset(self):
        collisions = 1
        while collisions:
            obs = self.env.reset()
            collisions = (abs(self.env.sim.data.get_sensor("B1_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B2_site_touch"))+
                        abs(self.env.sim.data.get_sensor("B3_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B4_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B5_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B6_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B7_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B8_site_touch")) +
                        abs(self.env.sim.data.get_sensor("B9_site_touch")) +
                        abs(self.env.sim.data.get_sensor("sw_site1_touch")) +
                        abs(self.env.sim.data.get_sensor("sw_site2_touch")) +
                        abs(self.env.sim.data.get_sensor("sw_site3_touch")) +
                        abs(self.env.sim.data.get_sensor("sw_site4_touch")))
        obs = self.observation(obs)
        return obs

    def render(self, mode):
        self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

    def step(self, action):
        self.action = action
        obs, rew, done, info = self.env.step(self.action)
        obs = self.observation(obs)
        return obs, rew, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec

    @property
    def n_actors(self):
        return self.env.metadata['n_actors']

    @property
    def t(self):
        return self.env.t
    
    @property
    def mjco_ts(self):
        return self.env.mjco_ts
    
    @property
    def n_substeps(self):
        return self.env.n_substeps

    @property
    def metadata(self):
        return self.env.metadata
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    @property
    def ts(self):
        return self.env.get_ts()
