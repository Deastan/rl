import gym

import tensorflow as tf
import numpy as np
from pyglet.gl import * # add for try
# from stable_baselines.ddpg.policies import MlpPolicy # Perceptrons!!!
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines import DDPG

# FetchSlide-v1

env = gym.make('FetchSlide-v1')
# env = DummyVecEnv([lambda: env])

env.reset()
env.render()


# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()