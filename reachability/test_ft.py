import numpy as np

from fastrack import FastTrack
from rm_tp import RmPlanner, RmTracker, RmRelative, Reach


tracker = RmTracker()
planner = RmPlanner()
relative = RmRelative()
reach = Reach()

ft = FastTrack(tracker, planner, relative, reach)
ft.run(100000)
