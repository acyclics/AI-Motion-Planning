import numpy as np
import time
from copy import deepcopy

from env import make_env


def test():
    env = make_env()
    obs = env.reset()
    # Test zero action
    zero_action = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    print(f"Before zero action = {obs}")
    new_obs, rew, done, _ = env.step(zero_action)
    print(f"After zero action = {new_obs}, is equal = {obs == new_obs}")
    obs = new_obs
    # Test dynamics
    dt = 0.002
    action = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    new_pos = deepcopy(obs)
    new_pos[0] = obs[0] + (1000*action[0, 0] * np.cos(obs[2]) + 1000*action[0, 1] * np.sin(obs[2])) * dt
    new_pos[1] = obs[1] + (1000*action[0, 1] * np.cos(obs[2]) - 1000*action[0, 0] * np.sin(obs[2])) * dt
    new_pos[2] = obs[2] + action[0, 2] * dt
    new_obs, rew, done, _ = env.step(action)
    print(f"Is done = {done}, Is dynamics correct = {new_pos == new_obs}")


def test_render():
    env = make_env(10000, visualize=True)
    obs = env.reset()
    #while True:
    #    env.render()
    zero_action = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #start_time = time.time()
    new_obs, rew, done, _ = env.step(zero_action)
    action = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    while True:
        new_obs, rew, done, _ = env.step(action)
        env.render()
        if done:
            break
    #print(time.time() - start_time)
    #while True:
    #    env.render()


#test()
test_render()
