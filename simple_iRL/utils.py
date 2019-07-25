'''
From https://docs.python.org/3/library/math.html

- math.exp(x):
    Return e raised to the power x, where e = 2.718281… is the base 
    of natural logarithms. This is usually more accurate 
    than math.e ** x or pow(math.e, x).

- math.log(x[, base]):
    With one argument, return the natural logarithm of x (to base e).
    With two arguments, return the logarithm of x to the given base, 
    calculated as log(x)/log(base).

- math.log1p(x):
    Return the natural logarithm of 1+x (base e). The result is 
    calculated in a way which is accurate for x near zero.

- math.pow(x, y):
    Return x raised to the power y. Exceptional cases follow Annex 
    ‘F’ of the C99 standard as far as possible. In particular, 
    pow(1.0, x) and pow(x, 0.0) always return 1.0, even when x is a 
    zero or a NaN. If both x and y are finite, x is negative, and y 
    is not an integer then pow(x, y) is undefined, and raises 
    ValueError.

    Unlike the built-in ** operator, math.pow() converts both its 
    arguments to type float. Use ** or the built-in pow() function 
    for computing exact integer powers.

-math.sqrt(x):
    Return the square root of x.

math.acos(x):
    Return the arc cosine of x, in radians.

math.asin(x):
    Return the arc sine of x, in radians.

math.atan(x):
    Return the arc tangent of x, in radians.

math.atan2(y, x):
    Return atan(y / x), in radians. The result is between -pi and pi. 
    The vector in the plane from the origin to point (x, y) makes this 
    angle with the positive X axis. The point of atan2() is that the 
    signs of both inputs are known to it, so it can compute the correct 
    quadrant for the angle. For example, atan(1) and atan2(1, 1) are 
    both pi/4, but atan2(-1, -1) is -3*pi/4.

math.cos(x):
    Return the cosine of x radians.

math.hypot(x, y):
    Return the Euclidean norm, sqrt(x*x + y*y). This is the length of 
    the vector from the origin to point (x, y).

math.sin(x):
    Return the sine of x radians.

math.tan(x):
    Return the tangent of x radians.
'''

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
