import os
import numpy as np
import xmltodict

from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.objs.obj import Obj
from collections import OrderedDict


class Battlefield(Obj):
    '''
    Battlefield() is essentially a model of the battlefield.
    It has no joints, so is essentially an immovable object.
    The XML for Battlefield is located in assets/xmls.
    '''
    @store_args
    def __init__(self):
        super(Battlefield, self).__init__()
        self.battlefield_xml = os.path.join(os.getcwd(), "environment", "assets", "xmls", "competition_area", "battlefield.xml")

    def generate(self, random_state, world_params, placement_size):
        top = OrderedDict(origin=(0, 0, 0), size=placement_size)
        self.placements = OrderedDict(top=top)
        self.size = np.array([placement_size[0], placement_size[1], 0.0])

    def generate_xml_dict(self):
        with open(self.battlefield_xml, "rb") as f:
            xmldict = xmltodict.parse(f)
        return xmldict['mujoco']
