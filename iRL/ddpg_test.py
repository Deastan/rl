import gym
import numpy as np
from pyglet.gl import * # add for try
from stable_baselines.ddpg.policies import MlpPolicy # Perceptrons!!!
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

# PPO2
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines import PPO2

import os
import matplotlib.pyplot as plt

# from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines import DDPG
# from stable_baselines.ddpg import AdaptiveParamNoiseSpec

import tensorflow as tf

#***************************** Global Params *********************************************
# Global params
train = True
predict = True
total_timesteps = 1000
best_mean_reward, n_steps = -np.inf, 0

# Create log dirfrom pyglet.gl import * # add for try
log_dir = "/tmp/gym/"
log_tensorboard = "/tmp/ddpg_MountainCarContinious_tensorboard/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_tensorboard, exist_ok=True)

# # Create and wrap the environment
# env = gym.make('LunarLanderContinuous-v2')
# # Logs will be saved in log_dir/monitor.csv
# env = Monitor(env, log_dir, allow_early_resets=True)

#***************************** Functions *********************************************

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=50)
    # # Truncate x
    # x = x[len(x) - len(y):]
    # # x = x[len(x) - len(y):]

    # fig = plt.figure(title)
    # plt.plot(x, y)
    # plt.xlabel('Number of Timesteps')
    # plt.ylabel('Rewards')
    # plt.title(title + " Smoothed")
    # plt.show()

#***************************** Function Main *********************************************
# sudo apt-get install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev freeglut3 xvfb libav-tools

def main():
    # Params
    global train, predict, log_dir, total_timesteps

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(tf.ConfigProto(allow_soft_placement=True))
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))


    env = gym.make('MountainCarContinuous-v0')
    # env = DummyVecEnv([lambda: env])
    # Create and wrap the environment
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('FetchReach-v0')
    # env = DummyVecEnv([lambda: env])
    # Create and wrap the environment
    # env = gym.make('LunarLanderContinuous-v2')
    
    # env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])



    # Logs will be saved in log_dir/monitor.csv
    # env = Monitor(env, log_dir, allow_early_resets=True)

    if train == True:
        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        # print(n_actions)
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        # tensorboard_log="/tmp/ddpg_MountainCarContinious_tensorboard/", 
        model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=True)
        # model = DDPG(MlpPolicy, env, verbose=1)
        # model.learn(total_timesteps=total_timesteps, callback=callback)
        model.learn(total_timesteps=total_timesteps)
        # model = PPO2(MlpPolicy, env, verbose=1)
        # model.learn(total_timesteps=total_timesteps)
        model.save("MountainCarContinuous")
        # model.save("ddpg_mountain")
        # model.save("ppo2_mountain")

        #Show we have finished with the learning
        print("Finisehd with learning")

        # plot_results(log_dir)
        del model # remove to demonstrate saving and loading
    # End if

    if predict == True:
        # model = DDPG.load("ddpg_mountain_40000") # Best one!
        model = DDPG.load("MountainCarContinuous")
        # model = DDPG.load("ddpg_mountain")
        # model = PPO2.load("ppo2_mountain")

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
        

#***************************** Main *********************************************
if __name__ == '__main__':
  try:
    main()
    print("Finished nicely")
  finally:
    print("Something wrong ?")
