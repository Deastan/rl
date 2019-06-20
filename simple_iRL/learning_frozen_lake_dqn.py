import random
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
   # Simplify the problem
   register(
      id="FrozenLakeNotSlippery-v0",
      entry_point='gym.envs.toy_text:FrozenLakeEnv',
      kwargs={'map_name': '4x4', 'is_slippery': False},
   )

   env = gym.make("FrozenLakeNotSlippery-v0")
   return env


def qlearning(env):
   state_size = env.observation_space.n
   action_size = env.action_space.n
   # print("Ovservation space: ", state_size)
   # print("Action space: ", action_size)

   # Use the Q-learning method

   # Create the q learnign table
   # matrix state_size by action_size 
   # Here it will be 16 x 4
   Q_table = np.zeros((state_size, action_size))
   # Table with the reward..
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
      print("************************")
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
         
         env.render()
         # time.sleep(0.1)

      EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
      # print(EPSILON)
      rewards.append(total_rewards)
      episode_list.append(episode)
   
   print(Q_table)


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

def main():
   env = init_env()
   # qlearning(env)

   # The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
   # tf.reset_default_graph()
   tf.compat.v1.reset_default_graph()

   #These lines establish the feed-forward part of the network used to choose actions
   inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
   W = tf.Variable(tf.random_uniform([16,4],0,0.01))
   Qout = tf.matmul(inputs1,W)
   predict = tf.argmax(Qout,1)

   #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
   nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
   loss = tf.reduce_sum(tf.square(nextQ - Qout))
   trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
   updateModel = trainer.minimize(loss)

if __name__ == '__main__':
   main()
   
   

# *********************************************************************
# https://www.reddit.com/r/learnmachinelearning/comments/9vasqm/review_my_code_dqn_for_gym_frozenlake/
# import gym
# import numpy as np 
# import random
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# def one_hot_state(state,state_space):
#     state_m=np.zeros((1,state_space))
#     state_m[0][state]=1
#     return state_m

# def experience_replay():
#     # Sample minibatch from the memory
#     minibatch = random.sample(memory, batch_size)
#     # Extract informations from each memory
#     for state, action, reward, next_state, done in minibatch:
#         # if done, make our target reward
#         target = reward
#         if not done:
#           # predict the future discounted reward
#           target = reward + gamma * \
#                    np.max(model.predict(next_state))
#         # make the agent to approximately map
#         # the current state to future discounted reward
#         # We'll call that target_f
#         target_f = model.predict(state)
#         target_f[0][action] = target
#         # Train the Neural Net with the state and target_f
#         model.fit(state, target_f, epochs=1, verbose=0)
        
# # 1. Parameters of Q-leanring    
# gamma = .9
# learning_rate=0.002
# episode = 5001
# capacity=64
# batch_size=32

# # Exploration parameters
# epsilon = 1.0           # Exploration rate
# max_epsilon = 1.0             # Exploration probability at start
# min_epsilon = 0.01            # Minimum exploration probability 
# decay_rate = 0.005             # Exponential decay rate for exploration prob

# # 2. Load Environment 
# env = gym.make("FrozenLake-v0")

# # env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# state_space=env.observation_space.n
# action_space=env.action_space.n

# #Neural network model for DQN
# model = Sequential()
# model.add(Dense(state_space, input_dim=state_space, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(action_space, activation='linear'))
# model.compile(loss='mse',
#               optimizer=Adam(lr=learning_rate))

# model.summary()


# reward_array=[]
# memory = deque([], maxlen=capacity)
# for i in range(episode):
#     state=env.reset()    
#     total_reward=0
#     done=False
#     while not done:        
#         state1_one_hot=one_hot_state(state,state_space)
#         # 3. Choose an action a in the current world state (s)
#         ## First we randomize a number
#         exp_exp_tradeoff = np.random.uniform()
        
#         ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
#         if exp_exp_tradeoff > epsilon:
#             action = np.argmax(model.predict(state1_one_hot))
#         # Else doing a random choice --> exploration
#         else:
#             action = env.action_space.sample()
        
#         #Training without experience replay
#         state2,reward,done,info=env.step(action)
#         state2_one_hot=one_hot_state(state2,state_space)
#         target = (reward + gamma *
#                       np.max(model.predict(state2_one_hot)))
                      
#         target_f=model.predict(state1_one_hot)
#         target_f[0][action]=target
#         model.fit(state1_one_hot,target_f,epochs=1,verbose=0)        
#         total_reward +=reward
                        
#         state=state2
        
#         #Training with experience replay
#         #appending to memory
#         memory.append((state1_one_hot, action, reward, state2_one_hot, done))
#         #experience replay
#     if i>batch_size:
#         experience_replay()
            
#     reward_array.append(total_reward)
#     # Reduce epsilon (because we need less and less exploration)
#     epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*i) 
    
#     if i % 10==0 and i !=0:
#         print('Episode {} Total Reward: {} Reward Rate {}'.format(i,total_reward,str(sum(reward_array)/i)))