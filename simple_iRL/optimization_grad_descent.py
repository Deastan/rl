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
def cal_cost_function(list_demos, np_theta):
    '''
    Calculates the cost function for the given parameters.
    
    Input:
        - Theta
        - List demos
    
    Ouput:
        - Cost function
    '''
    print("cal_cost_function()")
    # numberDemos = 5
    # numberDemos = len(list_demos)
    # np_theta = np.array([0.5, 0.1])
    cost = 0
    for i in range(0, len(list_demos)):
        m = len(list_demos[i]) - 1 # because of init 
        z_interm = 0
        beta = 0
        for j in range(1, m - 1):
            beta += np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
            z_interm += math.exp(-beta)
        
        z_theta = 1/m * z_interm
        print("z_theta: ", z_theta)
        cost += beta - math.log1p(z_theta) 
        
    return cost
    # end of cal_cost_function


# Gradient descent function
def grad_descent(list_demos, learning_rate=0.0001, iterations=10):
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
    print("grad_descent()")
    np_theta = 10*np.array([np.random.random_sample(), np.random.random_sample()])
    # print("grad_descent")
    # theta = np.random.randn(7, 1)
    print("theta random", np_theta)

    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
   

    loss_deriv_theta_1 = 0
    loss_deriv_theta_2 = 0

    for it in range(iterations):

        # Demos
        for i in range(0, len(list_demos)):
            m = len(list_demos[i]) - 1 # because of init 

            loss_deriv_theta_1_interm = 0
            loss_deriv_theta_2_interm = 0
            sumExpMinusThetaDotStates = 0
            for j in range(1, m-1):
                thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                sumExpMinusThetaDotStates += math.exp(-thetaDotState)
            print("sumExpMinusThetaDotStates: ", sumExpMinusThetaDotStates)
            # States of a trajectory
            for j in range(1, m - 1):

                thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                print("thetaDotState: ", thetaDotState)
                loss_deriv_theta_1_interm += list_demos[i][j][0] - list_demos[i][j][0] * math.exp(-thetaDotState) / sumExpMinusThetaDotStates 
                loss_deriv_theta_2_interm += list_demos[i][j][1] - list_demos[i][j][1] * math.exp(-thetaDotState) / sumExpMinusThetaDotStates 

            #end over the states
            
            loss_deriv_theta_1 += loss_deriv_theta_1_interm
            loss_deriv_theta_2 += loss_deriv_theta_2_interm
        #end over the demo

        np_theta[0] = np_theta[0] - learning_rate * loss_deriv_theta_1
        np_theta[1] = np_theta[1] - learning_rate * loss_deriv_theta_2        
        print("theta 1: ", np_theta[0], ", theta 2: ", np_theta[1])
        theta_history[it] = np_theta
        cost_history[it] = cal_cost_function(list_demos, np_theta)
        # loss_deriv_theta_1 = 1
        # loss_deriv_theta_1 = 1
        
        

    # end of the for

    return np_theta
    # end of grad_descend




def main():
    print("Main start!")

    # cal_cost_function()
    # grad_descent()name = name + ".pkl"
    name = "demos_frozenLake_test_3.pkl"
    with open(name, 'rb') as f:
        list_demos = pickle.load(f)
    
    print(list_demos)
    print("Nombre de demos:", len(list_demos))

    print("Nombre de point sur la trajectoire: ", len(list_demos[1]))
    print(list_demos[0][2][1])

    # print("Cost function: ", cal_cost_function(list_demos))

    print("Theta: ", grad_descent(list_demos))
    print("Main end!")

#***********************************
#       Main
#***********************************
if __name__ == '__main__':
    main()
    # try:
    #     main()
    #     print("Main function worked fine!")
    # except:
    #     print("ERROR: couldn't run the main function!")
    
