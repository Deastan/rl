# Jonathan Burkhard
# IRL
# TODO: Create more function here...

import random
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
# save part
import time, pickle, os

from gym.envs.registration import register

def take_action(action_1, env):
   new_state, reward_update, done, info = env.step(action_1)

   # reward function
   if new_state in  [5, 7, 11, 12]:
      reward_update = -1.0
      
   elif new_state in  [15]:
      # print("bim")
      reward_update = 1.0
      
   else:#if new_state == [0, 1, 2, 3, 4, 6, 8, 9, 10 , 13, 14]:
      reward_update = -0.1
   # print(new_state)
   return new_state, reward_update, done, info

def init_env():
   '''
   Init the env from gym

   Output: gym env
   '''
   # Remove the idea that the agent can iceskate...
   # Simplify the problem
   register(
      id="FrozenLakeNotSlippery-v0",
      entry_point='gym.envs.toy_text:FrozenLakeEnv',
      kwargs={'map_name': '4x4', 'is_slippery': False},
   )

   env = gym.make("FrozenLakeNotSlippery-v0")
   return env

def qlearning(env):
   '''
   Agent learns using q-learning
   '''
   state_size = env.observation_space.n
   action_size = env.action_space.n
    # Use the Q-learning method

   # Create the q learnign table
   # matrix state_size by action_size 
   # Here it will be 16 x 4
   Q_table = np.zeros((state_size, action_size))
   # Table with the reward and episode to allow me 
   # to plot a graph if needed
   rewards = []
   episode_list = []

   # Number of episode
   MAX_EPISODES = 15000
   print('Max episodes: ', MAX_EPISODES)
   # Parameters:
   ALPHA = 0.8
   GAMMA = 0.95   

   EPSILON = 1.0
   MAX_EPSILON = 1.0
   MIN_EPSILON = 0.01
   DECAY_RATE = 0.005
   step = 0

   # env.render(mode='ansi')

   for episode in range(MAX_EPISODES):
      state = env.reset()
      # print(step)
      step = 0
      
      # print(state)
      done = False
      total_rewards = 0
      # print("************************")
      while not done:
         # step 1
         if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
         else:
            action = np.argmax(Q_table[state, :])

         # step 2
         state_plus_one, R, done, info = take_action(action, env)
         # print("state n+1: ", state_plus_one, ", State: ", state,", Action: ", action, ", reward: ", R)
         # step 3
         q_predict = Q_table[state, action]

         if done:
            q_target = R
         else:
            q_target = R + GAMMA * np.max(Q_table[state_plus_one, :])
         Q_table[state, action] +=  ALPHA * (q_target - q_predict)

         state = state_plus_one
         # if state==15:
            # print("Objectif atteind")
            # print(done)
         total_rewards += R
         step += 1
         
         # env.render()
         # time.sleep(0.1)

      EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
      # print(EPSILON)
      rewards.append(total_rewards)
      episode_list.append(episode)
   
   print(Q_table)
   return Q_table, rewards, episode_list

def main():
   env = init_env()
   Q_table, rewards, episode_list = qlearning(env)

   # Save datas
   with open("frozenLake_qTable_15000.pkl", 'wb') as f:
      pickle.dump(Q_table, f)

   
   # print(rewards[MAX_EPISODES-1])
   # Plot Rewards
   plt.plot(episode_list, rewards)
   plt.xlabel('Episodes')
   plt.ylabel('Average Reward')
   plt.title('Average Reward vs Episodes')
   plt.savefig('rewards.jpg')     
   plt.show()
   # plt.close() 

if __name__ == '__main__':
   # main()
   try:
      main()
   except:
      print("ERROR: Cannot run the main")