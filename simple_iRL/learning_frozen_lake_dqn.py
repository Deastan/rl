import random
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# save part
import time, pickle, os

from gym.envs.registration import register

# Actions
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

# Understanding: np.identity(s)[i:i+1]
# Use np.identity or np.eye. You can try something like this 
# with your input i, and the array size s:
# np.identity(s)[i:i+1]
# For example, print(np.identity(5)[0:1]) will result:
# [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
# If you are using TensorFlow, you can use tf.one_hot: 
# https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#one_hot


def take_action(action_1, env):
   new_state, reward_update, done, info = env.step(action_1)

   # reward function
   if new_state in  [5, 7, 11, 12]:
      reward_update = -1.0
      
   elif new_state in  [15]:
      # print("bim")
      reward_update = 2.0
      
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


# Neural network with one hidden layer using tensorflow
def NNlearning(env):
   '''
   function which is a neural net with only one hidden layer.
   It multiply the weight matrix with the state vector to obtain the matrix of Q
   Then at the predict, caltulate the argmax of Q vector (four line) and obtain 
   predicted action
   '''
   # The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
   # tf.reset_default_graph()
   tf.compat.v1.reset_default_graph()


   # Implementing the network itself
   # These lines establish the feed-forward part of the network used to choose actions
   inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
   W = tf.Variable(tf.random_uniform([16,4],0,0.01))
   Qout = tf.matmul(inputs1,W)
   predict = tf.argmax(Qout,1)

   # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
   nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
   loss = tf.reduce_sum(tf.square(nextQ - Qout))
   trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
   updateModel = trainer.minimize(loss)

   # Training the network
   init = tf.global_variables_initializer()

   # Set learning parameters
   y = .99
   e = 0.1
   num_episodes = 100
   #create lists to contain total rewards and steps per episode
   jList = []
   rList = []
   with tf.Session() as sess:
      sess.run(init)
      for i in range(num_episodes):
         #Reset environment and get first new observation
         s = env.reset()
         rAll = 0
         d = False
         j = 0
         #The Q-Network
         while j < 99:
               j+=1
               print("Episode: ", i, ", step: ", j)
               #Choose an action by greedily (with e chance of random action) from the Q-network
               # inputs1:np.identity(16)[s:s+1]} : Put in the inputs1 a vector of 0 with a 1 on 
               # current state [s:s+1] => [5:6]: [0 0 0 0 1 0 0 ... 0] 
               a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
               print(a)
               if np.random.rand(1) < e:
                  a[0] = env.action_space.sample()
               #Get new state and reward from environment
               # s1,r,d,_ = env.step(a[0])
               s1,r,d,_ = take_action(a[0], env)
               #Obtain the Q' values by feeding the new state through our network
               Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
               # print("State: ", s1, ", action: ", a[0], ", Qout (actionPredictBy network): ", Q1)
               # print("AllQ:")
               # print(allQ)
               #Obtain maxQ' and set our target value for chosen action.
               maxQ1 = np.max(Q1)
               targetQ = allQ
               targetQ[0,a[0]] = r + y*maxQ1
               #Train our network using target and predicted Q values
               # print(inputs1)
               _, W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
               
               rAll += r
               s = s1
               env.render()
               if d == True:
                  #Reduce chance of random action as we train the model.
                  e = 1./((i/50) + 10)
                  break
         jList.append(j)
         rList.append(rAll)
      for k in range(0, 15):
         s = k
         a = sess.run([predict],feed_dict={inputs1:np.identity(16)[s:s+1]})
         print("For state: ", s, "It choose the action: ", a[0])
      # print(W1)
   # print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
   
   plt.plot(rList)
   plt.xlabel('Episodes')
   plt.ylabel('Reward')
   plt.title('Reward vs Episodes')
   plt.savefig('rewards.jpg')     
   plt.show()
   # test_pred = sess.run(prediction, feed_dict={s: 0})
   # Qout = tf.matmul(4,W)
   # predict = tf.argmax(Qout,1)
   # with tf.Session() as sess:
      # s = 4
      # a, allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
      # print(a)

# Neural network with 2 hidden layer using tensorflow
def DNNlearning(env):
   '''
   function which is a neural net with only one hidden layer.
   It multiply the weight matrix with the state vector to obtain the matrix of Q
   Then at the predict, caltulate the argmax of Q vector (four line) and obtain 
   predicted action
   '''
   # The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
   # tf.reset_default_graph()
   tf.compat.v1.reset_default_graph()


   # Implementing the network itself
   # These lines establish the feed-forward part of the network used to choose actions
   inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
   # W = tf.Variable(tf.random_uniform([16,4],0,0.01))
   W1 = tf.Variable(tf.random_uniform([16,16],0,0.01))
   # W2 = tf.Variable(tf.random_uniform([16,16],0,0.01))
   W2 = tf.Variable(tf.random_uniform([16,4],0,0.01))
   # W3 = tf.Variable(tf.random_uniform([16,4],0,0.01))
   Q12 = tf.matmul(inputs1,W1)
   # Q23 = tf.matmul(Q12,W2)
   # Qout = tf.matmul(inputs1,W)
   # Qout = tf.matmul(Q23,W3)
   Qout = tf.matmul(Q12,W2)
   predict = tf.argmax(Qout,1)

   # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
   nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
   loss = tf.reduce_sum(tf.square(nextQ - Qout))
   trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
   updateModel = trainer.minimize(loss)

   # Training the network
   init = tf.global_variables_initializer()

   # Set learning parameters
   y = .99
   e = 0.1
   num_episodes = 1000
   #create lists to contain total rewards and steps per episode
   jList = []
   rList = []
   with tf.Session() as sess:
      sess.run(init)
      for i in range(num_episodes):
         #Reset environment and get first new observation
         s = env.reset()
         rAll = 0
         d = False
         j = 0
         #The Q-Network
         while j < 15 and not d:
               j+=1
               # print("Episode: ", i, ", step: ", j)
               #Choose an action by greedily (with e chance of random action) from the Q-network
               # inputs1:np.identity(16)[s:s+1]} : Put in the inputs1 a vector of 0 with a 1 on 
               # current state [s:s+1] => [5:6]: [0 0 0 0 1 0 0 ... 0] 
               a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
               # print(a[0])
               if np.random.rand(1) < e:
                  a[0] = env.action_space.sample()
               #Get new state and reward from environment
               # s1,r,d,_ = env.step(a[0])
               s1,r,d,_ = take_action(a[0], env)
               #Obtain the Q' values by feeding the new state through our network
               Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
               # print("State: ", s1, ", action: ", a[0], ", Qout (actionPredictBy network): ", Q1)
               # print("AllQ:")
               # print(allQ)
               #Obtain maxQ' and set our target value for chosen action.
               maxQ1 = np.max(Q1)
               targetQ = allQ
               targetQ[0,a[0]] = r + y*maxQ1
               #Train our network using target and predicted Q values
               # print(inputs1)
               # _, W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
               _, W1plus, W2plus = sess.run([updateModel,W1, W2],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
               
               rAll += r
               s = s1
               # env.render()
               if d == True:
                  #Reduce chance of random action as we train the model.
                  e = 1./((i/50) + 10)
                  # e = 1 - i/num_episodes
                  # e = math.exp(-(i+1))
                  break
         jList.append(j)
         rList.append(rAll)
      for k in range(0, 15):
         s = k
         a = sess.run([predict],feed_dict={inputs1:np.identity(16)[s:s+1]})
         print("For state: ", s, "It choose the action: ", a[0])
      #Test algo
      print("***********Prediction***************")
      done = False
      t = 0
      state = env.reset()
      while not done:
         # env.render()
         # print("state is ", state)
         # action = choose_action(state, Q)  
         a = sess.run([predict],feed_dict={inputs1:np.identity(16)[state:state+1]})
         
         # print(a[0][0])
         # print(action)
         
         # state2, reward, done, info = env.step(action)  
         # state, reward, done, info = env.step(action) 
         new_state,r,done,_ = take_action(a[0][0], env)
         env.render()
         state = new_state
         t+=1
      # print(W1)
   # print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
   
   plt.plot(rList)
   plt.xlabel('Episodes')
   plt.ylabel('Reward')
   plt.title('Reward vs Episodes')
   plt.savefig('rewards.jpg')     
   plt.show()
   # test_pred = sess.run(prediction, feed_dict={s: 0})
   # Qout = tf.matmul(4,W)
   # predict = tf.argmax(Qout,1)
   # with tf.Session() as sess:
      # s = 4
      # a, allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
      # print(a)


from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)
# keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)


def choose_action(model, state, action_space, exploration_rate):
   if np.random.rand() < exploration_rate:
      return random.randrange(action_space)
   q_values = model.predict(state)
   return np.argmax(q_values[0])

# Neural network with 2 hidden layer using Keras
def DNNlearning_keras(env):
   '''
   function which is a neural net using Keras
   '''
 
   #PARAMS
   GAMMA = 0.95
   LEARNING_RATE = 0.001
   MEMORY_SIZE = 100
   BATCH_SIZE = 20
   EXPLORATION_MAX = 1.0
   EXPLORATION_MIN = 0.01
   EXPLORATION_DECAY = 0.995

   # Env params
   observation_space = 16
   # observation_space = env.observation_space
   action_space = 4

   exploration_rate = EXPLORATION_MAX

   # Model params
   model = Sequential()
   model.add(Dense(16, input_shape=(observation_space,), activation="relu"))
   model.add(Dense(16, activation="relu"))
   model.add(Dense(action_space, activation="linear"))
   model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

   # Experience replay:
   # memory = deque(maxlen=MEMORY_SIZE)

   episode_max = 10000
   done = False
   for i in range(episode_max):
      total_reward = 0
      j = 0
      state = env.reset()
      # state = np.reshape(state, [1, observation_space])
      state = np.identity(16)[state:state+1]
      done = False
      while j < 15 and not done: # step inside an episode
         j+=1
         print("[INFO]: episode: ", i, ", step: ", j)

         action = choose_action(model, state, action_space, exploration_rate)
         # print(action)
         new_state, reward, done, info = take_action(action, env)
         new_state = np.identity(16)[new_state:new_state+1]
         # (np.max(model.predict(new_state)))
         # q_target = reward + GAMMA * (np.max(model.predict(new_state)))
         if not done:
            q_target = reward + GAMMA * (np.amax(model.predict(new_state)))#(np.amax(model.predict(new_state)))
         else:
            #if done
            q_target = reward

         q_value = model.predict(state)
         q_value[0][action] = q_target
         model.fit(state, q_value, verbose=0)
         
         
         # if done:
         #    reward += reward
         
         state = new_state
         total_reward += reward
   
   #Test algo
   print("***********Prediction***************")
   prediction = True
   if prediction == True:
      done = False
      t = 0
      state = env.reset()
      state = np.identity(16)[state:state+1]
      while not done:
         #ACTION TO SET
         action = model.predict(state)
         new_state, reward, done, _ = take_action(np.argmax(action), env)
         new_state = np.identity(16)[new_state:new_state+1]         
         env.render()
         state = new_state
         t+=1
      # print(W1)
#end function

def save_forReplay(memory, state, action, reward, new_state, done):
   memory.append((state, action, reward, new_state, done))
   return memory

def experience_replay(model, memory, 
      BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA):
   if len(memory) < BATCH_SIZE:
      return
   batch = random.sample(memory, BATCH_SIZE)
   # print(batch)
   for state, action, reward, new_state, done in batch:
      # print("Inside replay")
      if not done:
         q_target = (reward + GAMMA * np.amax(model.predict(new_state)))
      else:
         q_target = reward

      q_values = model.predict(state)
      q_values[0][action] = q_target
      model.fit(state, q_values, verbose=1)#, callbacks=[tensorboard])

   exploration_rate *= EXPLORATION_DECAY
   exploration_rate = max(EXPLORATION_MIN, exploration_rate)
   # return model

# Neural network with 2 hidden layer using Keras + experience replay
def DDN_learning_keras_memoryReplay(env):
   '''
   function which is a neural net using Keras with memory replay
   '''
   EPISODE_MAX = 30
   #PARAMS
   GAMMA = 0.95
   LEARNING_RATE = 0.001
   MEMORY_SIZE = EPISODE_MAX
   BATCH_SIZE = 20
   EXPLORATION_MAX = 1.0
   EXPLORATION_MIN = 0.01
   EXPLORATION_DECAY = 0.995

   # Env params
   observation_space = 16
   # observation_space = env.observation_space
   action_space = 4

   exploration_rate = EXPLORATION_MAX

   # Model params
   model = Sequential()
   model.add(Dense(16, input_shape=(observation_space,), activation="relu"))
   model.add(Dense(16, activation="relu"))
   model.add(Dense(action_space, activation="linear"))
   model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

   # Experience replay:
   memory = deque(maxlen=MEMORY_SIZE)

   episode_max = EPISODE_MAX
   done = False
   for i in range(episode_max):
      total_reward = 0
      j = 0
      state = env.reset()
      # state = np.reshape(state, [1, observation_space])
      state = np.identity(16)[state:state+1]
      done = False
      while j < 15 and not done: # step inside an episode
         j+=1
         print("[INFO]: episode: ", i, ", step: ", j)

         action = choose_action(model, state, action_space, exploration_rate)
         # print(action)
         new_state, reward, done, info = take_action(action, env)
         new_state = np.identity(16)[new_state:new_state+1]

         # Momory replay
         # memory = save_forReplay(memory, state, action, reward, new_state, done)
         save_forReplay(memory, state, action, reward, new_state, done)
         experience_replay(model, memory, 
            BATCH_SIZE, exploration_rate, EXPLORATION_DECAY, EXPLORATION_MIN, GAMMA)

         state = new_state
         total_reward += reward
   
   #Test algo
   print("***********Prediction***************")
   prediction = True
   if prediction == True:
      done = False
      t = 0
      state = env.reset()
      state = np.identity(16)[state:state+1]
      while ((not done) and t < 15):
         #ACTION TO SET
         action = model.predict(state)
         new_state, reward, done, _ = take_action(np.argmax(action), env)
         new_state = np.identity(16)[new_state:new_state+1]         
         env.render()
         state = new_state
         t+=1
      # print(W1)
#end function

def main():
   env = init_env()
   # qlearning(env)
   # NNlearning(env)
   # DNNlearning(env)
   # DNNlearning_keras(env)
   DDN_learning_keras_memoryReplay(env) #converged one time at 100 epochs
   


   #enf of the main function
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