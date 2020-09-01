import numpy as np

from navigation import Navigation


def test():
    nav = Navigation()
    actions, neglogp, entropy, value, xyyaw_mean, xyyaw_logstd = nav(np.zeros([1, 10]))
    print(actions)


test()
