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
def cal_cost_function():
    '''
    Calculates the cost function for the given parameters.
    
    Input:
        - Theta
    
    Ouput:
        - Cost function
    '''
    print("cal_cost_function")
    cost = 1
    # end of cal_cost_function


# Gradient descent function
def grad_descent(learning_rate=0.01, iterations=100):
    '''
    Calculates iteratively the gradient

    Pseudo-code:
    The gradient tells us the incline or slope of the cost function. Hence, to minimize the cost function, we move in the direction opposite to the gradient.

    1) Initialize the weights W randomly.
    2) Calculate the gradients G of cost function w.r.t parameters. This is done using partial differentiation: G = ∂J(W)/∂W. The value of the gradient G depends on the inputs, the current values of the model parameters, and the cost function. You might need to revisit the topic of differentiation if you are calculating the gradient by hand.
    3) Update the weights by an amount proportional to G, i.e. W = W - ηG
    4) Repeat until the cost J(w) stops reducing, or some other pre-defined termination criteria is met.

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
    grad_descent()
    
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
    
