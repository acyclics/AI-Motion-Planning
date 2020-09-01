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


class CollisionWrapper(gym.Wrapper):
    '''
        Adds collision detection and punishment.
        Also provides info for agents' distance to obstacle.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = 4
        # Collision force threshold
        self.threshold = 0.0
        
    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get indexes of qvel for movement debuff
        self.agent_qvel_idxs = [qvel_idxs_from_joint_prefix(sim, f'agent{i}') for i in range(self.n_agents)]
        # Get indexes of qpos of agents
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        # Get indexes of obstacles pos
        self.obstacles_id = [
            sim.model.body_name2id("B1"),
            sim.model.body_name2id("B2"),
            sim.model.body_name2id("B3"),
            sim.model.body_name2id("B4"),
            sim.model.body_name2id("B5"),
            sim.model.body_name2id("B6"),
            sim.model.body_name2id("B7"),
            sim.model.body_name2id("B8"),
            sim.model.body_name2id("B9")
        ]
        return self.observation(obs)

    def observation(self, obs):
        return obs

    def collision_detection(self, info):
        sim = self.unwrapped.sim
        for idx in range(0, 2):
            agent_collision = (
                abs(sim.data.get_sensor(f'agent{idx}:wheel_fl_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:wheel_fr_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:wheel_bl_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:wheel_br_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:armor_r_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:armor_b_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:armor_f_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:armor_l_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:bar_f_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:bar_b_touch')) +
                abs(sim.data.get_sensor(f'agent{idx}:body_touch'))
            )
            if agent_collision != 0:
                return False, -1000
        return False, 0

    def movement_restriction(self, action):
        for i in range(2, 4):
            action['action_movement'][i] = np.array([0, 0, 0])
        return action
    
    def step(self, action):
        # Restrict action for stationary agents
        action = self.movement_restriction(action)
        obs, rew, done, info = self.env.step(action)
        colli_done, colli_rew = self.collision_detection(info)
        done = done | colli_done
        rew += colli_rew
        return self.observation(obs), rew, done, info
