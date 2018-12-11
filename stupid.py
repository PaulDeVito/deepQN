import math
import random
import numpy as np 
import dqn
import yahtzee as yt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.distributions import Categorical

from collections import namedtuple



# a single transition in our enviroment
Transition = namedtuple('Transition', 
					('state', 'action', 'next_state', 'reward'))


def generate_input(dice, rolls_left):
	inputs = np.zeros((6,6))
	for i in range(5):
		for j in range(dice[i]):
			inputs[j][i] = 1

	for j in range(rolls_left):
		inputs[j][5] = 1
	return inputs

def choose_action_greedy(dice, model):
	print(dice)
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	distribution = model(state)
	print(distribution)
	output, index = distribution.max(0)
	print(index)
	print(output)
	print(math.log(output))
	# output, index = torch.max(distribution)

	return index, math.log(output)

def choose_action_probabilistic(dice, model):
	# print(dice)
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	categories = Categorical(model(state))
	action = categories.sample()

	# print(action)
	# print(categories.log_prob(action))

	return action, categories.log_prob(action)

def calculate_mean_log_prob(n):
	action_space = 2 ** n
	average_prob = 1./action_space
	return math.log(average_prob)


num_dice = 2
gamma = 0.99

l = dqn.Stupid(num_dice,2 ** num_dice)
mean_log_prob = calculate_mean_log_prob(num_dice)
l.train()






num_games = 100000000
env = yt.mini_environment(num_dice,2)

running_average = []
bad_rolls = 0
num_yahtzees = 0
loss_history = []

print("starting...")
for game in range(num_games):
	env.reset()
	# l.action_history = Variable(torch.Tensor(), requires_grad=True)
	# for i in range(rolls_allowed):
	prev_score = env.points
		# print("old score: ", (prev_score))
	save, log = choose_action_probabilistic(env.dice, l)
	# print(save)
	# print(list(l.parameters()))
		# if (l.action_history.dim() != 0):
			# l.action_history = torch.cat([l.action_history,log])
		# else:
			# l.action_history = log

		# action_history.append(log)

	env.step_simple(save)
	score = env.points
		# print("new score: " + str(score))
		# if(score == 50):
			# num_yahtzees += 1
	reward = score - prev_score
	# print("reward: ", reward)
	# print("log: ", log)

	loss = -(log - mean_log_prob) * reward
	# print("loss: ", loss)
	# loss = Variable(loss, requires_grad=True)

	l.optimizer.zero_grad()
	loss.backward()
	l.optimizer.step()

	# print(list(l.parameters())[0].grad)



	running_average.append(env.points)
	bad_rolls += env.num_max_dice_rerolled
	# print(num_yahtzees)
	if (game%5000 == 0):
		print("average: ", str(sum(running_average)/5000), "  bad rolls: ", bad_rolls/5000)
		running_average = []
		bad_rolls = 0
