import random
import time
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import utils

# This is the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D

import sys, termios, tty, os, time #method 3
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# save part
import time, pickle, os

# save the data
def save(list_theta, name="theta_default_name"):
    saved = False
    try:
        name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(list_theta, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = True
    except:
        print("Couldn't save the file .pkl")
    return saved

# Load the data
# TODO
def load(name="demos_default_name"):
    name = name + ".pkl"
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except:
        print("Couldn't load ", name, "!")
        empty = []
        return empty

def state_to_xy(state):
    # print("New state is: ", state)
    # x direction is 0 1 2 3
    # y direction is 0 4 8 12 
    x = (state) % 4 + 1 
    y = int(math.modf((state) / 4)[1]) + 1
    # print("Cartesian ( ", x, ", ", y, ")", end='\n')
    return x, y

def xy_to_state(x, y):
    # print("New state is: ", state)
    # x direction is 0 1 2 3
    # y direction is 0 4 8 12 
    # x = (state) % 4 + 1 
    # y = int(math.modf((state) / 4)[1]) + 1
    # print("Cartesian ( ", x, ", ", y, ")", end='\n')
    state = 0

    state = int((x-1)+4*(y-1))
    return state

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    # sys.stdout.write('Loading video: ')
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
