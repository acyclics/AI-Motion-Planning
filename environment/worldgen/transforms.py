import numpy as np
from collections import OrderedDict

'''
    Transforms are functions which modify the world in-place.
    They can be used to add, remove or change specific attributes, tags, etc.
'''

def closure_transform(closure):
    '''
        Call closure on every OrderedDict.
        This transform is usually not used directly, it is just called internally
        by other transforms.
    '''
    def recursion(xml_dict):
        closure(xml_dict)
        for key in list(xml_dict.keys()):
            values = xml_dict[key]
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if isinstance(value, OrderedDict):
                    recursion(value)
    return recursion


def set_geom_attr_transform(name, value):
    ''' Sets an attribute to a specific value on all geoms '''
    return set_node_attr_transform('geom', name, value)


def set_node_attr_transform(nodename, attrname, value):
    '''
        Sets an attribute to a specific value on every node of the specified type (e.g. geoms).
    '''
    def fun(xml_dict):
        def closure(node):
            if nodename in node:
                for child in node[nodename]:
                    child["@" + attrname] = value
        return closure_transform(closure)(xml_dict)
    return fun

def add_weld_equality_constraint_transform(name, body_name1, body_name2):
    '''
        Creates a weld constraint that maintains relative position and orientation between
        two objects
    '''
    def fun(xml_dict):
        if 'equality' not in xml_dict:
            xml_dict['equality'] = OrderedDict()
            xml_dict['equality']['weld'] = []
        constraint = OrderedDict()
        constraint['@name'] = name
        constraint['@body1'] = body_name1
        constraint['@body2'] = body_name2
        constraint['@active'] = False
        xml_dict['equality']['weld'].append(constraint)
        return xml_dict

    return fun

def set_joint_damping_transform(damping, joint_name):
    ''' Set joints damping to a single value.
        Args:
            damping (float): damping to set
            joint_name (string): partial name of joint. Any joint with joint_name
                as a substring will be affected.
    '''
    def closure(node):
        for joint in node.get('joint', []):
            if joint_name in joint['@name']:
                joint['@damping'] = damping
    return closure_transform(closure)

def remove_hinge_axis_transform(axis):
    ''' Removes specific hinge axis from the body. '''
    def fun(xml_dict):
        def closure(node):
            if 'joint' in node:
                node["joint"] = [j for j in node["joint"]
                                 if j["@type"] != "hinge"
                                 or np.linalg.norm(j["@axis"] - axis) >= 1e-5]
        return closure_transform(closure)(xml_dict)
    return fun
