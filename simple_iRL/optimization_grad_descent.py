# Jonathan Burkhard
# IRL
# help link :
#               - https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
#               - https://www.freecodecamp.org/news/understanding-gradient-descent-the-most-popular-ml-algorithm-a66c0d97307f/
#               - https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html


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

# Return the calculated cost function
def cal_cost_function(list_demos):
    '''
    Calculates the cost function for the given parameters.
    
    Input:
        - Theta
        - List demos
    
    Ouput:
        - Cost function
    '''
    print("cal_cost_function")
    # numberDemos = 5
    numberDemos = len(list_demos)
    # demos = 5
    for i in range(list_demos):
        m = len(list_demos[i]) - 1 # because of init 
        z_interm = 0
        for j in range(1, m + 1):
            z_interm = 0

        z_theta = 1/m
        -math.log1p(z_theta)
    cost = 1
    # end of cal_cost_function


# Gradient descent function
def grad_descent(learning_rate=0.01, iterations=100):
    '''
    Calculates iteratively the gradient

    Pseudo-code:
    The gradient tells us the incline or slope of the cost function. Hence, to 
    minimize the cost function, we move in the direction opposite to the gradient.

    1) Initialize the weights W randomly.
    2) Calculate the gradients G of cost function w.r.t parameters. This is done using partial 
        differentiation: G = ∂J(W)/∂W. The value of the gradient G depends on the inputs, the 
        current values of the model parameters, and the cost function. You might need to revisit 
        the topic of differentiation if you are calculating the gradient by hand.
    3) Update the weights by an amount proportional to G, i.e. W = W - ηG
    4) Repeat until the cost J(w) stops reducing, or some other pre-defined 
        termination criteria is met.

    Input:

    Output: 
    '''

    print("grad_descent")
    theta = np.random.randn(7, 1)
    print(theta)


    for i in range(iterations):
        print("inside loops: ", i)
        

    # end of the for

    return theta
    # end of grad_descend




def main():
    print("Main start!")

    # cal_cost_function()
    # grad_descent()name = name + ".pkl"
    name = "demos_frozenLake_test.pkl"
    with open(name, 'rb') as f:
        list_demos = pickle.load(f)
    
    print(list_demos)
    print("Nombre de demos:", len(list_demos))

    print("Nombre de point sur la trajectoire: ", len(list_demos[1]))
    print(list_demos[0][2][0])

    print("Main end!")

#***********************************
#       Main
#***********************************
if __name__ == '__main__':
    try:
        main()
        print("Main function worked fine!")
    except:
        print("ERROR: couldn't run the main function!")
    
