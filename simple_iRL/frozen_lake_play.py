import gym
import numpy as np
import time
import pickle, os
from gym.envs.registration import register

register(
        id="FrozenLakeNotSlippery-v0",
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
    )
# env = gym.make('FrozenLake-v0')
env = gym.make("FrozenLakeNotSlippery-v0")

# with open("frozenLake_qTable.pkl", 'rb') as f:
# 	Q = pickle.load(f)

def choose_action(state, Q):
    # LEFT = 0
    # DOWN = 1
    # RIGHT = 2
    # UP = 3
	action1 = np.argmax(Q[state, :])

    # print(action)

	return action1

if __name__ == '__main__':
    with open("frozenLake_qTable_15000.pkl", 'rb') as f:
	    Q = pickle.load(f)
    
    print(Q)
    # done = False
    # start
    for episode in range(2):
        done = False
        state = env.reset()
        print("*** Episode: ", episode)
        t = 0
        # env.render()
        while not done:
            # env.render()
            print("state is ", state)
            action = choose_action(state, Q)  
            print(action)
            env.render()
            # state2, reward, done, info = env.step(action)  
            state, reward, done, info = env.step(action) 
            t+=1
            # env.render()
            # state = state2
            
            # if done:
            #     # print(state)
            #     print("Done!")
            #     break

            # time.sleep(0.5)

    # os.system('clear')