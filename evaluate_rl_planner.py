import numpy as np

from rl.env import make_env
from rl.navigation import Navigation


env = make_env(visualize=True)
obs = env.reset()

nav = Navigation(1, False)
nav.call_build()

weights = np.load("./rl/data/nav_model.npy", allow_pickle=True)
nav.set_weights(weights)

while True:
    actions, neglogp, entropy, value, logits = nav(np.expand_dims(obs, axis=0))
    actions = {k: (v[0] - 10) / 10 for k, v in actions.items()}
    agent_actions = np.array([[actions['x1'], actions['y1'], actions['w1']],
                                [actions['x2'], actions['y2'], actions['w2']]])
    obs, rewards, dones, infos = env.step(agent_actions)
    env.render()
    if dones:
        obs = env.reset()
