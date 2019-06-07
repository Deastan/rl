import gym
from pyglet.gl import *
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


#************************* PPO2 ********************************************
env = gym.make('CarRacing-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
print("Training finished")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
print("finished")
env.close()
#************************* DDPG ********************************************
# import gym
# import numpy as np

# # from stable_baselines.ddpg.policies import MlpPolicy
# # from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines import DDPG

# env = gym.make('MountainCarContinuous-v0')
# env = DummyVecEnv([lambda: env])

# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
# model.learn(total_timesteps=400000)
# model.save("ddpg_mountain")

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_mountain")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


#************************* Simple programm ********************************************
# Simple code to run the race
# env.reset()

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()