import numpy as np
from copy import deepcopy

from reachability.rm_tp import RmPlanner, RmTracker, RmRelative, Reach


def make_planner_env():
    from rl.env import make_env
    env = make_env()
    return env


def make_tracker_env():
    from environment.envs.icra import make_env
    from environment.envhandler import EnvHandler
    env = EnvHandler(make_env())
    return env


class Simulator():

    def __init__(self):
        # environments
        self.simul_env = make_tracker_env()
        self.planner_env = make_planner_env()
        self.eps = 1e-5
        self.dt = 0.01
        # models
        self.planner = RmPlanner()
        self.relative = RmRelative()
        self.reach = Reach()

    def get_goals(self):
        goals = [
            [np.random.uniform() * 8080, np.random.uniform() * 4480],
            [np.random.uniform() * 8080, np.random.uniform() * 4480]
        ]
        return goals

    def get_planner_control(self, p, goal):
        # TEMPORARY
        p[2] = 0
        return p

    def get_relative_state(self, s, pnext):
        return self.relative.state(s, pnext)
    
    def simul_to_states(self, obs, u1, u2):
        s1 = [0.0 for _ in range(5)]
        s2 = [0.0 for _ in range(5)]

        s1[0] = obs[0] / 1000
        s1[1] += u1[0] * self.dt
        s1[2] = obs[1] / 1000
        s1[3] += u1[1] * self.dt
        s1[4] = obs[2]

        s2[0] = obs[6] / 1000
        s2[1] += u2[0] * self.dt
        s2[2] = obs[7] / 1000
        s2[3] += u2[1] * self.dt
        s2[4] = obs[8]

        return s1, s2

    def get_next_states(self, s1, s2, u1, u2):
        agent_actions = {'action_movement': np.array([ [s1[1] / 2.0, s1[3] / 3.0, u1[2] / 6.0],
                                                       [s2[1] / 2.0, s2[3] / 3.0, u2[2] / 6.0],
                                                       [0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0] ])}

        for _ in range(int(self.dt / 0.002)):
            obs, rewards, dones, infos = self.simul_env.step(agent_actions)

        s1, s2 = self.simul_to_states(obs, u1, u2)

        return s1, s2

    def simulate(self):
        while True:
            obs = self.simul_env.reset()

            s1, s2 = self.simul_to_states(obs, [0, 0, 0], [0, 0, 0])
            s = [s1, s2]

            p = [self.planner.project(s[0]), self.planner.project(s[1])]            

            goals = [
                [obs[18], obs[19]],
                [obs[20], obs[21]]
            ]

            while (abs(p[0][0] - goals[0][0]) >= self.eps and
                   abs(p[0][1] - goals[0][1]) >= self.eps and
                   abs(p[1][0] - goals[1][0]) >= self.eps and
                   abs(p[1][1] - goals[1][1]) >= self.eps):

                pnext = [self.get_planner_control(p[i], goals[i]) for i in range(2)]
                rnext = [self.get_relative_state(s[i], pnext[i]) for i in range(2)]
                rnext = [self.reach.to_grid_index(rnext[i]) for i in range(2)]

                u = [
                    self.relative.optControl(self.reach.get_derivs(rnext[i]), rnext[i], s[i])
                    if self.reach.check_on_boundary(rnext[i]) else self.reach.control(s[i], pnext[i])
                    for i in range(2)  
                ]
                """
                if self.reach.check_on_boundary(rnext):
                    deriv = self.reach.get_derivs(rnext)
                    u = self.relative.optControl(deriv, rnext, s)
                else:
                    u = self.reach.control(s, pnext)
                """
                self.simul_env.render('human')

                # New states
                s[0], s[1] = self.get_next_states(s[0], s[1], u[0], u[1])
                p[0], p[1] = self.planner.project(s[0]), self.planner.project(s[1])


simulator = Simulator()
simulator.simulate()
