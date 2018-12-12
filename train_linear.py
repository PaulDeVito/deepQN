import math
import random
import time
import csv
import numpy as np 
import dqn
import yahtzee as yt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.distributions import Categorical


def choose_action_greedy(dice, model):
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	distribution = model(state)
	output, index = distribution.max(0)



	return index, math.log(output)

def choose_action_probabilistic(dice, model):
	# print(dice)
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	categories = Categorical(model(state))
	action = categories.sample()

	return action, categories.log_prob(action).unsqueeze(0), categories.logits.mean()


num_dice = 5
action_space = 2^num_dice
state_space = 252
gamma = 0.99

l = dqn.Linear(num_dice,2 ** num_dice)
#l.train()




num_games = 1000000
env = yt.mini_environment(num_dice, 6)
epoch_length = 10000
rolls_allowed = 2

long_running_average = []
long_loss_history = []
long_num_yahtzees = []
long_running_time_elapsed = []
running_average = []
num_yahtzees = 0
loss_history = []
start_time = time.time()

for game in range(num_games):
	env.reset()
	reward_history = []
	l.action_history = torch.Tensor()
	for i in range(rolls_allowed):
		prev_score = env.points
		save, log, avg = choose_action_probabilistic(env.dice, l)

		if (l.action_history.dim() != 0):
			l.action_history = torch.cat([l.action_history,-(log - avg)])
		else:
			l.action_history = -(log - avg)


		env.step_simple(save)
		score = env.points
		if(score == 50):
			num_yahtzees += 1
		reward = score - prev_score
		reward_history.append(reward)


	# calculate loss using bellman equation
	q_i_plus_one = 0
	ltr = []
	for i in range(rolls_allowed-1, -1, -1):
		q_i_plus_one = reward_history[i] + gamma * q_i_plus_one
		ltr.insert(0,q_i_plus_one)

	ltr = torch.FloatTensor(ltr)
	loss = torch.sum(l.action_history*ltr)

	# update weights
	l.optimizer.zero_grad()
	loss.backward()
	l.optimizer.step()

	running_average.append(env.points)
	loss_history.append(loss)
	if (game%epoch_length == 0):
		points = sum(running_average)/epoch_length
		avg_loss = (sum(loss_history)/epoch_length).item()
		time_elapsed = time.time() - start_time
		print("Game ", str(game), " -------------------------------------------------------------------------")
		print("average: ", points, "  Yahtzees: ", num_yahtzees, "   Loss: ", avg_loss, "   Time: ", time_elapsed)
		long_running_average.append(points)
		long_loss_history.append(avg_loss)
		long_num_yahtzees.append(num_yahtzees)
		long_running_time_elapsed.append(time_elapsed)
		running_average = []
		loss_history = []
		num_yahtzees = 0

print("Long running averate:")
print(long_running_average)
print("Loss history")
print(long_loss_history)
print("Number of Yahtzees")
print(long_num_yahtzees)
print("Elapsed time")
print(long_running_time_elapsed)

with open('output.csv', mode='w') as out_file:
	out_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	for i in range(len(long_running_average)):
		out_writer.writerow([i*epoch_length,long_running_average[i],long_loss_history[i],long_num_yahtzees[i]])