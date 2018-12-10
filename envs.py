import gym
print(type(gym.envs.registry.all()))
env_list = list(gym.envs.registry.all())
print(type(env_list))

for environment in env_list:
	print(environment)

# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
    # env.render()
    # env.step(env.action_space.sample()) # take a random action