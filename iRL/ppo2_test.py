
# import gym
# from pyglet.gl import *
# # from stable_baselines.common.policies import MlpPolicy
# # from stable_baselines.common.vec_env import SubprocVecEnv
# # from stable_baselines.common.vec_env import DummyVecEnv
# # from stable_baselines import PPO2
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

# # multiprocess environment
# # n_cpu = 4
# # env = gym.make('CartPole-v1')
# env = gym.make('LunarLander-v2')
# # env = gym.make('LunarLanderContinuous-v2')
# # env = gym.make('MountainCarContinuous-v0')
# # env = SubprocVecEnv([lambda: env])# for i in range(n_cpu)])
# env = DummyVecEnv([lambda: env])
# train = True

# # if train == True:
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=5000)
#   # model.save("LunarLanderContinuous_500000")

#   # del model # remove to demonstrate saving and loading

# # model = PPO2(MlpPolicy, env, verbose=1)
# # model = PPO2.load("LunarLanderContinuous_500000")

# # # Enjoy trained agent
# obs = env.reset()
# while True:
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   env.render()

##########################################################################################################################
# Work for low number of timestep and not every world...

import gym
from pyglet.gl import * # add for try
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
print("End of script")


###########################################################################################################################

# import numpy as np
# from pyglet.gl import * # add for try

# import os
# import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines import PPO2

# #***************************** Global Params *********************************************
# # Global params
# train = True
# predict = True
# total_timesteps = 4000
# best_mean_reward, n_steps = -np.inf, 0

# # Create log dir
# # log_dir = "/tmp/gym/"
# # os.makedirs(log_dir, exist_ok=True)

# # # Create and wrap the environment
# # env = gym.make('LunarLanderContinuous-v2')
# # # Logs will be saved in log_dir/monitor.csv
# # env = Monitor(env, log_dir, allow_early_resets=True)

# #***************************** Functions *********************************************



# #***************************** Function Main *********************************************
# def main():
#     # Params
#     # global train, predict, log_dir, total_timesteps
#     global train, predict, total_timesteps

#     # config = tf.ConfigProto()
#     # config.gpu_options.allow_growth = True
#     # config = tf.ConfigProto(allow_soft_placement=True)
#     # sess = tf.Session(tf.ConfigProto(allow_soft_placement=True))
#     # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))


#     # env = gym.make('MountainCarContinuous-v0')
#     # env = DummyVecEnv([lambda: env])
#     # Create and wrap the environment
#     # env = gym.make('MountainCarContinuous-v0')
#     # env = gym.make('CarRacing-v0')
#     # env = DummyVecEnv([lambda: env])
#     # Create and wrap the environment
#     # DDPG
#     # env = gym.make('LunarLanderContinuous-v2')
#     # env = Monitor(env, log_dir, allow_early_resets=True)
#     # env = DummyVecEnv([lambda: env])
#     # multiprocess environment
#     n_cpu = 4
#     env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])


#     # Logs will be saved in log_dir/monitor.csv
#     # env = Monitor(env, log_dir, allow_early_resets=True)

#     if train == True:
#         # the noise objects for DDPG
#         # n_actions = env.action_space.shape[-1]
#         # param_noise = None
#         # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

#         # model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=True)
#         # model = DDPG(MlpPolicy, env, verbose=1)
#         # model.learn(total_timesteps=total_timesteps, callback=callback)
#         # model.learn(total_timesteps=total_timesteps)
#         # model = PPO2(MlpPolicy, env, verbose=1)
#         # model.learn(total_timesteps=total_timesteps)
#         # model.save("ppo2_CarRacing")
#         # model.save("ddpg_mountain")
#         # model.save("ppo2_cartpole")
#         model = PPO2(MlpPolicy, env, verbose=1)
#         model.learn(total_timesteps=total_timesteps)
#         model.save("ppo2_cartpole_biiiim")

#         #Show we have finished with the learning
#         print("Finisehd with learning")

#         # plot_results(log_dir)
#         del model # remove to demonstrate saving and loading
#     # End if

#     if predict == True:
#         print("Start with working mode")
#         # model = DDPG.load("ddpg_mountain_40000") # Best one!
#         # model = DDPG.load("CartPole")
#         # model = DDPG.load("ddpg_mountain")
#         # model = PPO2.load("ppo2_mountain")
#         # model = PPO2.load("ppo2_cartpole")
#         model = PPO2.load("ppo2_cartpole_biiiim")

#         obs = env.reset()
#         while True:
#             action, _states = model.predict(obs)
#             obs, rewards, dones, info = env.step(action)
#             env.render()

# #***************************** Main *********************************************
# if __name__ == '__main__':
#   try:
#     main()
#   finally:
#     print("Finished nicely")
