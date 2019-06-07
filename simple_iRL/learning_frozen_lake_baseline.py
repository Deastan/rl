import random
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

# RL ALGO
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# save part
import time, pickle, os

from gym.envs.registration import register

###################################################################
#     Functions 
###################################################################

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

##################################################################
#     Main function 
###################################################################

def main():

   # Remove the idea that the agent can iceskate...
   # Simplify the problem
   register(
      id="FrozenLakeNotSlippery-v0",
      entry_point='gym.envs.toy_text:FrozenLakeEnv',
      kwargs={'map_name': '4x4', 'is_slippery': False},
   )

   env = gym.make("FrozenLakeNotSlippery-v0")
   # print(env)
   # state_size = env.observation_space.n
   # action_size = env.action_space.n
   # print("Ovservation space: ", state_size)
   # print("Action space: ", action_size)

   # We will use the Q-learning method
   model = DQN(MlpPolicy, env, verbose=1)
   model.learn(total_timesteps=25000)

   print("Saving model in .pkl")
   model.save("deepq_FrozenLakeNotSlippery")

   obs = env.reset()
   while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

   # act.save("cartpole_model.pkl")

##################################################################
#     Main  
###################################################################

if __name__ == '__main__':
   try:
      main()
      print("Program works normally")
   except KeyboardInterrupt:
        print("Killed by user using his her keyboard")
        sys.exit(0)

