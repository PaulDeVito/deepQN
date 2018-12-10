import gym
import math
import random
import numpy as np

from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make('CartPole-v0')


for i_episode in range(10):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action) # take a random action
		if done:
			print("DONE")
			break