import random
import time
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import sys, termios, tty, os, time #method 3
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# save part
import time, pickle, os

from gym.envs.registration import register

# Listen the event and the keyboard and return the keboard caracter
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# save the data
def save(list_demos, name="demos_default_name"):
    saved = False
    try:
        name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(list_demos, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = True
    except:
        print("Couldn't save the file .pkl")
    return saved

# Load the data
# TODO
def load(name="demos_default_name"):
    name = name + ".pkl"
    with open(name, 'rb') as f:
        return pickle.load(f)

def state_to_xy(state):
    # print("New state is: ", state)
    x = (state) % 4 + 1 
    y = int(math.modf((state) / 4)[1]) + 1
    # print("Cartesian ( ", x, ", ", y, ")", end='\n')
    return x, y


def main():
    # Remove the idea that the agent can iceskate...
    # Simplify the problem
    register(
        id="FrozenLakeNotSlippery-v0",
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
    )

    env = gym.make("FrozenLakeNotSlippery-v0")
    print(env)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    # print("Ovservation space: ", state_size)
    # print("Action space: ", action_size)

    # print("Next action ? ")
    quit = False
    restart = False
    bool_save =  False
    bool_load = False
    done = False
    action = -1
    i = 0
    
    list_demos = []
    # list_states = []
    np_oneDemo = np.array([[0, 0, 0]])
    print(np_oneDemo)
    # list_oneDemo = []
    # list_actions = []

    while not quit:
        list_states = []
        list_actions = []
        nomber_of_move = 0
        done = False
        state = env.reset()
        env.render()
        while not done:
            # LEFT = 0
            # DOWN = 1
            # RIGHT = 2
            # UP = 3

            # method 3
            button_delay = 0.1
 
            # while True:
            char = getch()
        
            if (char == "p"):
                print("Stop!")
                exit(0)
        
            if (char == "a"):
                # print("Left pressed")
                action = 0
                time.sleep(button_delay)
        
            elif (char == "d"):
                # print("Right pressed")
                action = 2
                time.sleep(button_delay)
        
            elif (char == "w"):
                # print("Up pressed")
                action = 3
                time.sleep(button_delay)
        
            elif (char == "s"):
                # print("Down pressed")
                action = 1
                time.sleep(button_delay)
            
            elif (char == "e"):
                # print("Down pressed")
                # action = 1
                bool_save = True
                time.sleep(button_delay)
                
            elif (char == "l"):
                # print("Down pressed")
                # action = 1
                bool_load = True
                time.sleep(button_delay)
        
            elif (char == "q"):
                # print("Number 1 pressed")
                quit = True
                done = True
                print("You quit!")
                
                time.sleep(button_delay)
 
            if action !=-1:
                new_state, reward, done, info = env.step(action)
                nomber_of_move += 1
                print(done)
                x, y = state_to_xy(new_state)
                np_oneDemo = np.append(np_oneDemo, [[x, y, action]], axis=0)
                
                state = new_state
                env.render()
            action = -1
        # End off steps
        
        if quit != True:
            # we don't want the last one
            
            list_demos.append(np_oneDemo)
        action = -1
        i+=1

    # print("The demos are: ")
    # print(list_demos)
    # print("Finish the game")
    env.close()

    if bool_save == True:
        save(list_demos)

    if bool_load == True:
        print(load())

#***********************************
#       Main
#***********************************
if __name__ == '__main__':
    main()
    
