# import matplotlib as plt
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt

# This is the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D


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

def function2D(i):
    
    e = 1 / ((i/50) + 10)
    return e

def plot2D():
    '''
    Plot the cose function
    Input: list_demos which is the recorded demos from an expert
    Output: 3D plot
    '''

    # plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-200.0, 10000.0, 0.05)
    # X = np.meshgrid(x)
    zs = np.array([function2D(np.array([x])) for x in zip(np.ravel(x))])
    Z = zs.reshape(x.shape)

    ax.plot(x, Z)

    ax.set_xlabel('X Label')
    ax.set_zlabel('Z Label')

    plt.show()


def main():
    plot2D()

    # plt.plot(rList)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.title('Reward vs Episodes')
    # plt.savefig('rewards.jpg')     
    # plt.show()


if __name__ == '__main__':
   main()
   