# import gym

# env = gym.make('CartPole-v0')
# env.reset()

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

# env.close()
import gym

print("Start")
env = gym.make('CartPole-v0')
env.reset()

for i in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()
