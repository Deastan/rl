# Jonathan Burkhard
# IRL
# help link :
#               - https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
#               - https://www.freecodecamp.org/news/understanding-gradient-descent-the-most-popular-ml-algorithm-a66c0d97307f/
#               - https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html


import random
import time
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
from itertools import product

# This is the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D

import sys, termios, tty, os, time #method 3
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# save part
import time, pickle, os


def next_state_calculation(state, action):
    if action == 0:
        if(state !=0 and state != 4 and state != 8 and state != 12):
            next_state = state - 1
        else:
            next_state = state
    elif action == 1:
        if(state != 12 and state != 13 and state != 14 and state != 15):
            next_state = state + 4
        else:
            next_state = state
    elif action == 2:
        if(state != 3 and state != 7 and state != 11 and state != 15):
            next_state = state + 1
        else:
            next_state = state
    elif action == 3:
        if(state != 0 and state != 1 and state != 2 and state != 3):
            next_state = state - 4
        else:
            next_state = state
    else:# verify but normally should go here!
        next_state = state
        print("[ ERROR]: The action isn't right!")

    return next_state


def features_states(x, y):
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
    return np_features

def rewardFunction(np_theta, x, y):
    '''
    Return the reward for the theta, and the state
    '''
    np_features = features_states(x,y)
    return np.dot(np_theta, np_features)



def expected_state_visitation_frequency(list_demos, np_theta):
    '''

    '''
    number_states = 16
    number_actions = 4
    # initializate for testing
    # np_theta = 10*np.array([np.random.random_sample(), np.random.random_sample()])
    # np_features_for_size = features_states(1, 1)
    # np_theta = np.zeros(np_features_for_size.size)
    # for i in range(np_features_for_size.size):
    #     np_theta[i] = 10 * np.random.random_sample()

    
    z_partition = np.zeros((number_states, number_actions), dtype=float)
    z_partition_init = np.ones((number_states, number_actions), dtype=float)
    for state in range(number_states):
        for action in range(number_actions):
            # print("here")

            x, y = utils.state_to_xy(state)
            next_state = next_state_calculation(state, action)
            z_partition[state, action] += transition_probability(next_state, action, state) * math.exp(rewardFunction(np_theta, x, y)) * z_partition_init[next_state, action]
    
    z_partition_state = np.zeros(number_states, dtype=float)
    for state in range(number_states): 
        for action in range(number_actions):
            z_partition_state[state] += z_partition[state, action]
    
    # Proba of action i to j knowing state i
    policy = np.zeros((number_actions, number_states), dtype=float)
    for state in range(number_states):
        for action in range(number_actions):
            policy[action, state] = z_partition[state, action]/z_partition_state[state]


    
    # expected_state_visitation_frequency
    # proba_init_state_start = 1
    # proba_init_state_other = 0
    iterations = 1000 #(N)  in the paper
    iterations = len(list_demos)
    D = np.zeros((number_states, iterations), dtype = float)

    number_of_start_at_state = np.zeros(number_states, dtype=float)
    for trajectory in list_demos:
        x = trajectory[0][0]
        y = trajectory[0][1]
        state = utils.xy_to_state(x, y)
        number_of_start_at_state[state] += 1
        
        # D[] += 1/len(list_demos)
    p_start = number_of_start_at_state/len(list_demos)
    # print(p_start)
    
    for i in range(iterations):
        D[:, i] = p_start

    # for a state inside a selected demo:
    for i in range(iterations-1):
        # for state in range(states):
        for state_prime, action, state in product(range(number_states), range(number_actions), range(number_states)):
            # for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            # next_state = next_state_calculation(state, action)
            D[state, i+1] += D[state_prime, i]*transition_probability(state, action, state_prime)*policy[action, state_prime]

    D_vector = np.zeros(number_states, dtype=float) # (1 x number of state)
    for state in range(number_states):
        for i in range(iterations-1):
            D_vector[state] += D[state, i+1]

    return D_vector
    # return expected_svf.sum(axis=1)

def transition_probability(next_state, action, state):
    '''
    Return the probability to be on the next state knowing state and action
    input: 
        next_state: state after the action
        action: action can be 0, 1, 2, 3 respectively left, down, right, up
        state: state before the action
    output:
        probability to be on the next state knowing state and action

    '''
    proba = 0

    if action == 0:
        if((state-1) == next_state and state != 4 and state != 8 and state != 12):
            proba = 1
        else:
            proba = 0
    elif action == 1:
        if((state+4) == next_state and state != 12 and state != 13 and state != 14 and state != 15):
            proba = 1
        else:
            proba = 0
    elif action == 2:
        if((state+1) == next_state and state != 3 and state != 7 and state != 11 and state != 15):
            proba = 1
        else:
            proba = 0
    elif action == 3:
        if((state-4) == next_state and state != 0 and state != 1 and state != 2 and state != 3):
            proba = 1
        else:
            proba = 0
    else:# verify but normally should go here!
        proba = 0
        print("[ ERROR]: The action isn't right!")

    return proba

def maxEnt_optimization(list_demos):
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
    # parameters
    iterations = 1000
    learning_rate = 0.001

    np_features_for_size = features_states(1, 1)
    np_theta = np.zeros(np_features_for_size.size)
    for i in range(np_features_for_size.size):
        np_theta[i] = 10 * np.random.random_sample()

    # Compute f_tild
    f_tild = np.zeros((np_features_for_size.size), dtype=float)
    for demo in list_demos:
        for trajectory in demo:
            # f_tild = 1/len(list_demos) * features_states()
            # state = utils.xy_to_state(trajectory[0], trajectory[1])
            x = trajectory[0]
            y = trajectory[1]
            f_tild += features_states(x,y)

            
    number_states = 16
    # f_matrix = np.zeros(5, dtype=float) 
    # for i in range(16):
    #     x, y = utils.state_to_xy(i)
    #     f_matrix.append(features_states(x, y))
    # x, y = utils.state_to_xy(0)
    # f_matrix = [features_states(x, y)]
    # Try 2
    # f_matrix = np.zeros(number_states, np_features_for_size)
    # for i in range(0, 15):
    #     x, y = utils.state_to_xy(i)
    #     # np.concatenate(f_matrix, [features_states(x, y)])
    #     # print(features_states(x, y))
    #     # f_matrix = np.append(f_matrix, [features_states(x, y)], axis=0)
    #     f_matrix[i]= features_states(x, y)
    # try3
    f_matrix = np.zeros((number_states, np_features_for_size.size), dtype=float)
    for i in range(0, 15):
        x, y = utils.state_to_xy(i)
        # np.concatenate(f_matrix, [features_states(x, y)])
        # print(features_states(x, y))
        # f_matrix = np.append(f_matrix, [features_states(x, y)], axis=0)
        f_matrix[i, :]= features_states(x, y)
    

    for i in range(iterations):
        D_vector = expected_state_visitation_frequency(list_demos, np_theta)

        grad_loss_function = f_tild - f_matrix.T.dot(D_vector)

        np_theta = np_theta + learning_rate * grad_loss_function
        
        utils.progress(i, iterations, status=str(np_theta))

    return np_theta

    

def main():
    print("Start")
    # Open the demos files
    # list of demos:
    # list_demos[demos][trajectory][state x or state y or action]
    list_demos = utils.load(name="demos_frozenLake_test_3")
    # print(list_demos)
    # print(list_demos[0][0][1])
    # print(utils.state_to_xy(1, 4))
    # Check transition proba:
    # print(transition_probability(15, 2, 15))

    # check expeceted visitation
    # expected_state_visitation_frequency()

    # print(next_state_calculation(13, 1))

    # print(expected_state_visitation_frequency(list_demos))


    print(maxEnt_optimization(list_demos))
    print("Main end!")

#***********************************
#       Main
#***********************************
if __name__ == '__main__':
    main()
    # f_matrix = np.zeros(5, dtype=float) 
    
        # f_matrix.append(features_states(x, y)]), axis=1)
    # f_matrix
    # x, y = utils.state_to_xy(2)
    # print(features_states(x, y))
        
    # f_matrix = np.zeros((16, 5), dtype=float)
    # for i in range(0, 15):
    #     x, y = utils.state_to_xy(i)
    #     # np.concatenate(f_matrix, [features_states(x, y)])
    #     # print(features_states(x, y))
    #     # f_matrix = np.append(f_matrix, [features_states(x, y)], axis=0)
    #     f_matrix[i, :]= features_states(x, y)
    # print(f_matrix)

