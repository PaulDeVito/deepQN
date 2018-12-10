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
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	output, index = torch.max(model(state))

	return index

def choose_action_probabilistic(dice, model):
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	categories = Categorical(model(state))
	action = categories.sample()

	return action, categories.log_prob(action)

num_dice = 5
action_space = 2^num_dice
state_space = 252
gamma = 0.99

l = dqn.Linear(5,32)
l.train()





num_games = 1000000000
env = yt.mini_environment()
# some much needed testing. like why is there no yahtzee

rolls_allowed = 10
running_average = []
num_yahtzees = 0

for game in range(num_games):
	env.reset()
	reward_history = []
	action_history = []
	for i in range(rolls_allowed):
		prev_score = env.points
		save, log = choose_action_probabilistic(env.dice, l)

		# if (l.action_history.dim() != 0):
			# l.action_history = torch.cat([l.action_history,log])
		# else:
			# l.action_history = log

		action_history.append(log)

		env.step(save)
		score = env.points
		if(score == 50):
			num_yahtzees += 1
		reward = score - prev_score
		reward_history.append(reward)


	# calculate loss using bellman equation
	q_i_plus_one = 0
	ltr = np.zeros(rolls_allowed)
	reward_history.append(0)
	# print(reward_history)
	for i in range(rolls_allowed-1, -1, -1):
		# print(long_term_rewards)
		q_i_plus_one = reward_history[i] + gamma * q_i_plus_one
		ltr[i] = q_i_plus_one

	ltr = (ltr - ltr.mean()) / (ltr.std() + np.finfo(np.float32).eps)

	l.action_history = torch.tensor(action_history).type(torch.FloatTensor)
	l.reward_history = torch.from_numpy(ltr).type(torch.FloatTensor)

	loss = torch.sum(torch.mul(l.action_history,l.reward_history).mul(-1), -1)

	loss = Variable(loss, requires_grad=True)

	l.optimizer.zero_grad()
	loss.backward()
	l.optimizer.step()


	running_average.append(env.points)
	# print(num_yahtzees)
	if (game%5000 == 0):
		print("average: ", str(sum(running_average)/5000), "  Yahtzees: ", num_yahtzees)
		running_average = []
		num_yahtzees = 0
