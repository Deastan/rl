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
    # print("cal_cost_function()")
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
        # print("z_theta: ", z_theta)
        cost += beta - math.log1p(z_theta) 
        
    return cost
    # end of cal_cost_function

def cal_cost_function_v2(list_demos, np_theta):
    '''
    Calculates the cost function for the given parameters.
    Use features and not states
    
    Input:
        - Theta
        - List demos
    
    Ouput:
        - Cost function
    '''
    # print("cal_cost_function()")
    # numberDemos = 5
    # numberDemos = len(list_demos)
    # np_theta = np.array([0.5, 0.1])
    cost = 0
    # features = []
    for i in range(0, len(list_demos)):
        m = len(list_demos[i]) - 1 # because of init 
        z_interm = 0
        beta = 0
        for j in range(1, m - 1):
            np_features = features_states(list_demos[i][j][0], list_demos[i][j][1])
            # beta += np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
            beta += np.dot(np_theta, np_features).astype(np.float32)
            z_interm += math.exp(-beta)
        
        z_theta = 1/m * z_interm
        # print("z_theta: ", z_theta)
        cost += beta - math.log1p(z_theta) 
        
    return cost
    # end of cal_cost_function


# Gradient descent function
def grad_descent(list_demos, learning_rate=0.00001, iterations=25):
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

    Inputs:
            list_demos:
            learning_rate=0.00001:
            iterations=25:

    Output: 
            np_theta
            theta_history
            cost_history
    '''
    # print("grad_descent()")
    
    # Init params
    
    np_theta = 10*np.array([np.random.random_sample(), np.random.random_sample()])
    # np_theta = np.array([20,-20])
    # print("grad_descent")
    # theta = np.random.randn(7, 1)
    print("theta random", np_theta)

    cost_history = np.zeros(1)
    cost_history = np.append(cost_history, 0)
    cost_history = np.append(cost_history, cal_cost_function(list_demos, np_theta))
    # theta_history = np.zeros((2,2))
    theta_history = np_theta
    

    loss_deriv_theta_1 = 0
    loss_deriv_theta_2 = 0

    it = 0
    for it in range(iterations):
    # while cost_history[-1]-cost_history[-2] <= 5:

        # Demos
        for i in range(0, len(list_demos)):
            m = len(list_demos[i]) - 1 # because of init 

            loss_deriv_theta_1_interm = 0
            loss_deriv_theta_2_interm = 0
            sumExpMinusThetaDotStates = 0
            for j in range(1, m-1):
                thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                sumExpMinusThetaDotStates += math.exp(-thetaDotState)
            # print("sumExpMinusThetaDotStates: ", sumExpMinusThetaDotStates)
            # States of a trajectory
            for j in range(1, m - 1):

                thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                # print("thetaDotState: ", thetaDotState)
                loss_deriv_theta_1_interm += list_demos[i][j][0] - list_demos[i][j][0] * math.exp(-thetaDotState) / sumExpMinusThetaDotStates 
                loss_deriv_theta_2_interm += list_demos[i][j][1] - list_demos[i][j][1] * math.exp(-thetaDotState) / sumExpMinusThetaDotStates 

            #end over the states
            
            loss_deriv_theta_1 += loss_deriv_theta_1_interm
            loss_deriv_theta_2 += loss_deriv_theta_2_interm
        #end over the demo

        np_theta[0] = np_theta[0] - learning_rate * loss_deriv_theta_1
        np_theta[1] = np_theta[1] - learning_rate * loss_deriv_theta_2        
        print("theta 1: ", np_theta[0], ", theta 2: ", np_theta[1])
        # theta_history[it] = np_theta
        # cost_history[it] = cal_cost_function(list_demos, np_theta)
        theta_history = np.append(theta_history, np_theta)#, axis=0)
        cost_history = np.append(cost_history, cal_cost_function(list_demos, np_theta))#, axis=0)
        print("Cost: ", cost_history[-1])
        # loss_deriv_theta_1 = 1
        # loss_deriv_theta_1 = 1
        it+=1
        
    # plot(cost_history)
    # end of the for
    print("Number of iteration: ", it)
    return np_theta, theta_history, cost_history
    # end of grad_descend

# Gradient descent function v2
# Use features function
def grad_descent_v2(list_demos, learning_rate=0.0001, iterations=2000):
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

    Inputs:
            list_demos:
            learning_rate=0.00001:
            iterations=25:

    Output: 
            np_theta
            theta_history
            cost_history
    '''
    # print("grad_descent()")
    
    # Init params
    np_features_for_size = features_states(1, 1)
    np_theta = np.zeros(np_features_for_size.size)
    for i in range(np_features_for_size.size):
        np_theta[i] = 10 * np.random.random_sample()
    # np_theta = 1*np.array([
    #     np.random.random_sample()
    #     , np.random.random_sample()
    #     , np.random.random_sample()
    #     , np.random.random_sample()
    #     , np.random.random_sample()
    #     # , np.random.random_sample()
    # ])
    # np_theta = np.array([20,-20])
    # print("grad_descent")
    # theta = np.random.randn(7, 1)
    print("theta random", np_theta)

    cost_history = np.zeros(1)
    cost_history = np.append(cost_history, 0)
    cost_history = np.append(cost_history, cal_cost_function_v2(list_demos, np_theta))
    # theta_history = np.zeros((2,2))
    theta_history = np_theta
    

    loss_deriv_theta_1 = 0
    loss_deriv_theta_2 = 0
    loss_deriv_theta = [0] * np_theta.size
    it = 0
    for it in range(iterations):
    # while cost_history[-1]-cost_history[-2] <= 5:

        # Demos
        for i in range(0, len(list_demos)):
            m = len(list_demos[i]) - 1 # because of init 

            loss_deriv_theta_1_interm = 0
            loss_deriv_theta_2_interm = 0
            loss_deriv_theta_interm = [0] * np_theta.size
            sumExpMinusThetaDotStates = 0
            for j in range(1, m-1):
                np_features = features_states(list_demos[i][j][0], list_demos[i][j][1])
                thetaDotState = np.dot(np_theta, np_features)
                # thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                sumExpMinusThetaDotStates += math.exp(- thetaDotState.astype(np.float32))
            # print("sumExpMinusThetaDotStates: ", sumExpMinusThetaDotStates)
            # States of a trajectory
            for j in range(1, m - 1):
                np_features = features_states(list_demos[i][j][0], list_demos[i][j][1])
                # thetaDotState = np_theta[0]*list_demos[i][j][0] + np_theta[1]*list_demos[i][j][1]
                thetaDotState = np.dot(np_theta, np_features)
                # print("thetaDotState: ", thetaDotState)
                # loss_deriv_theta_1_interm += list_demos[i][j][0] - list_demos[i][j][0] * math.exp(- thetaDotState.astype(np.float32)) / sumExpMinusThetaDotStates
                # loss_deriv_theta_1_interm += np_features[0] - np_features[0] * math.exp(- thetaDotState.astype(np.float32)) / sumExpMinusThetaDotStates  
                # loss_deriv_theta_2_interm += np_features[1] - np_features[1] * math.exp(- thetaDotState.astype(np.float32)) / sumExpMinusThetaDotStates 
                
                for i in range(np_theta.size):
                    loss_deriv_theta_interm[i] += np_features[i] - np_features[i] * math.exp(- thetaDotState.astype(np.float32)) / sumExpMinusThetaDotStates  

            #end over the states
            
            # loss_deriv_theta_1 += loss_deriv_theta_1_interm
            # loss_deriv_theta_2 += loss_deriv_theta_2_interm
            for i in range(np_theta.size):
                loss_deriv_theta[i] += loss_deriv_theta_interm[i]
        #end over the demo

        # np_theta[0] = np_theta[0] - learning_rate * loss_deriv_theta_1
        # np_theta[1] = np_theta[1] - learning_rate * loss_deriv_theta_2   
        for i in range(np_theta.size):
            np_theta[i] = np_theta[i] + learning_rate * loss_deriv_theta[i]

        # print("theta 1: ", np_theta[0], ", theta 2: ", np_theta[1])
        # theta_history[it] = np_theta
        # cost_history[it] = cal_cost_function(list_demos, np_theta)
        theta_history = np.append(theta_history, np_theta)#, axis=0)
        cost_history = np.append(cost_history, cal_cost_function_v2(list_demos, np_theta))#, axis=0)
        # print("Cost: ", cost_history[-1])
        # loss_deriv_theta_1 = 1
        # loss_deriv_theta_1 = 1
        it+=1
        
    # plot(cost_history)
    # end of the for
    # print("Number of iteration: ", it)
    return np_theta, theta_history, cost_history
    # end of grad_descend_v2    

def features_states(x,y):
    '''
    return a vector of features of the state

    Inputs:
            x
            y
    Output:
            vector of features of the state
    '''
    # list_demos[i][j][0]
    # list_demos[i][j][1]

    # Try to add goal
    int_start = 0
    if x == 1 and y == 1:
        int_start = 1
    int_goal = 0
    if x == 4 and y == 4:
        int_goal = 1
    int_hole = 0
    if (x == 1 and y == 2) or (x == 4 and y == 2) or (x == 1 and y == 4):
        # "4x4": [
        # "SFFF",
        # "FHFH",
        # "FFFH",
        # "HFFG"]
        int_hole = 1
    int_not_hole = 0
    if not ((x == 2 and y == 2) or (x == 4 and y == 2) or (x == 1 and y == 4)):
        # "4x4": [
        # "SFFF",
        # "FHFH",
        # "FFFH",
        # "HFFG"]
        int_not_hole = 1
    # int_proximity = 0
    # if (x == 3 and y == 4):
    #     int_proximity = 1

    np_features = np.zeros(2)
    np_features[0] = x
    np_features[1] = y
    np_features = np.append(np_features, int_start)
    np_features = np.append(np_features, int_goal)
    np_features = np.append(np_features, int_not_hole)
    # np_features = np.append(np_features, int_hole)
    # np_features = np.append(np_features, x*x)
    # np_features = np.append(np_features, y*y)
    # np_features = np.append(np_features, x*y)
    # np_features = np.append(np_features, int_proximity)
    return np_features

def features_states_v0(x,y):
    '''
    return a vector of features of the state

    Inputs:
            x
            y
    Output:
            vector of features of the state
    '''
    # list_demos[i][j][0]
    # list_demos[i][j][1]

    # Try to add goal
    int_start = 0
    if x == 1 and y == 1:
        int_start = 1
    int_goal = 0
    if x == 4 and y == 4:
        int_goal = 1
    # int_hole = 0
    # if (x == 1 and y == 2) or (x == 4 and y == 2) or (x == 1 and y == 4):
    #     # "4x4": [
    #     # "SFFF",
    #     # "FHFH",
    #     # "FFFH",
    #     # "HFFG"]
    #     int_hole = 1
    int_not_hole = 0
    if not ((x == 2 and y == 2) or (x == 4 and y == 2) or (x == 1 and y == 4)):
        # "4x4": [
        # "SFFF",
        # "FHFH",
        # "FFFH",
        # "HFFG"]
        int_not_hole = 1
    # int_proximity = 0
    # if (x == 3 and y == 4):
    #     int_proximity = 1

    np_features = np.zeros(2)
    np_features[0] = x
    np_features[1] = y
    np_features = np.append(np_features, int_start)
    np_features = np.append(np_features, int_goal)
    np_features = np.append(np_features, int_not_hole)
    # np_features = np.append(np_features, int_proximity)
    return np_features

def features_states_simple(x,y):
    '''
    return a vector of features of the state

    Inputs:
            x
            y
    Output:
            vector of features of the state
    '''
    # list_demos[i][j][0]
    # list_demos[i][j][1]
    np_features = np.zeros(2)
    np_features[0] = x
    np_features[1] = y

    return np_features



def plot(cost_history):
    '''
    Cost history. 
    '''

    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost')
    plt.savefig('costPlot.jpg')     
    plt.show()

def plot3D(list_demos):
    '''
    Plot the cose function
    Input: list_demos which is the recorded demos from an expert
    Output: 3D plot
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 5.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([cal_cost_function(list_demos, np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot3D_v2(list_demos):
    '''
    Plot the cose function
    Input: list_demos which is the recorded demos from an expert
    Output: 3D plot
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 5.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([cal_cost_function_v2(list_demos, np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def rewardFunction(np_theta, x, y):
    '''
    Return the reward for the theta, and the state
    '''
    return x*np_theta[0]+y*np_theta[1]

def rewardFunction_v2(np_theta, x, y):
    '''
    Return the reward for the theta, and the state
    '''
    np_features = features_states(x,y)
    return np.dot(np_theta, np_features)


def rewardGrid(np_theta):
    '''
    Show what is the reward for each state in the grid
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    ys = [1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2 ,3]
    zs = [rewardFunction(np_theta, x, y) for x,y in zip(xs,ys)]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel(np_theta)

    plt.show()

def rewardGrid_v2(np_theta):
    '''
    Show what is the reward for each state in the grid
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    ys = [1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2 ,3]
    zs = [rewardFunction_v2(np_theta, x, y) for x,y in zip(xs,ys)]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel(np_theta)

    plt.show()

def main():
    # Open the demos files
    # grad_descent()name = name + ".pkl"
    name = "demos_frozenLake_test_3.pkl"
    with open(name, 'rb') as f:
        list_demos = pickle.load(f)
    
    print(list_demos)
    print("Nombre de demos:", len(list_demos))

    print("Nombre de point sur la trajectoire: ", len(list_demos[1]))
    print(list_demos[0][2][1])

    # np_theta = np.array([10, 10])
    # print("Cost function: ", cal_cost_function(list_demos, np_theta))
    # print("Cost function_v2: ", cal_cost_function_v2(list_demos, np_theta))
    # cal_cost_function_v2

    np_theta, theta_history, cost_history = grad_descent_v2(list_demos)
    print("Theta: ", np_theta)
    print("Cost: ", cost_history[-1], ", length: ", len(cost_history))

    # print("Cost History: ", cost_history)
    
    # Show the cost function
    # plot3D(list_demos)
    # plot3D_v2(list_demos)

    #show the gridReward
    # rewardGrid(np_theta)
    # rewardGrid_v2(np_theta)

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
    #     # Maybe save the files, lists etc... here
    #     # save(list_theta, name="theta_default_name")
    #     print("ERROR: couldn't run the main function!")
    
