import numpy as np
from copy import deepcopy


class Tracker():

    def __init__(self):
        self.size = 6

    def dynamics(self, x, u):
        raise NotImplementedError

    def optControl(self, x, rnext):
        raise NotImplementedError
    

class Planner():

    def __init__(self):
        self.size = 3

    def dynamics(self, x, u):
        raise NotImplementedError

    def control(self, x, goal):
        raise NotImplementedError


class Relative():

    def __init__(self):
        self.size = 3

    def dynamics(self, x, u):
        raise NotImplementedError

    def control(self, x, goal):
        raise NotImplementedError
