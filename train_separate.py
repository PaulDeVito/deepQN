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
	distribution = model(state)
	output, index = distribution.max(0)



	return index, math.log(output)



num_dice = 5
action_space = 2^num_dice
state_space = 252
gamma = 0.99

sm = nn.Softmax(dim=0)

def update_action_history(l, log, avg):
	if (l.action_history.dim() != 0):
			l.action_history = torch.cat([l.action_history,-(log - avg)])
	else:
		l.action_history = -(log - avg)

def choose_average_action_probabilistic(dice, models):
	state = Variable(torch.tensor(dice).type(torch.FloatTensor))
	p_vals = []
	for i in range(2 ** num_dice):
		p_vals.append(0.)
	p_vals = torch.tensor(p_vals)
	c_dists = []
	for m in models:
		output = m(state)
		p_vals += output
		c_dists.append(Categorical(output))
		
	# pick averaged action
	p_vals = sm(p_vals)
	categories = Categorical(p_vals)
	action = categories.sample()

	# update action history for each model
	for i in range(len(models)):
		c = c_dists[i]
		l = models[i]
		log = c.log_prob(action).unsqueeze(0)
		avg = c.logits.mean()
		update_action_history(l, log, avg)
		

	return action

def update_weights(l, reward_history):
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

	return loss

l_kind = dqn.Linear(num_dice,2 ** num_dice)
l_two_kind = dqn.Linear(num_dice,2 ** num_dice)
l_straight = dqn.Linear(num_dice,2 ** num_dice)
l_flush = dqn.Linear(num_dice,2 ** num_dice)
#l.train()

models = [l_kind, l_two_kind, l_straight, l_flush]


num_games = 10000000
env = yt.full_environment("all")
# some much needed testing. like why is there no yahtzee
epoch_length = 1000
rolls_allowed = 3

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
	reward_history_kind = []
	reward_history_two_kind = []
	reward_history_straight = []
	reward_history_flush = []
	for l in models:
		l.action_history = torch.Tensor()

	for i in range(rolls_allowed):
		prev_score_kind = env.score("kind")
		prev_score_two_kind = env.score("two_kind")
		prev_score_straight = env.score("straight")
		prev_score_flush = env.score("flush")
		save = choose_average_action_probabilistic(env.dice, models)

		env.step(save)

		reward_history_kind.append(env.score("kind") - prev_score_kind)
		reward_history_two_kind.append(env.score("two_kind") - prev_score_two_kind)
		reward_history_straight.append(env.score("straight") - prev_score_straight)
		reward_history_flush.append(env.score("flush") - prev_score_flush)


		real_score = env.score("all")
		if(real_score == 50):
			num_yahtzees += 1

	
	# print("first loop")
	loss1 = update_weights(l_kind, reward_history_kind)
	loss2 = update_weights(l_two_kind, reward_history_two_kind)
	loss3 = update_weights(l_straight, reward_history_straight)
	loss4 = update_weights(l_flush, reward_history_flush)



	# loss_kind = calculate_loss(l_kind, reward_history_kind)
	# loss_two_kind = calculate_loss(l_two_kind, reward_history_two_kind)
	# loss_straight = calculate_loss(l_straight, reward_history_straight)
	# loss_flush = calculate_loss(l_flush, reward_history_flush)
# 
	# l_kind.optimizer.zero_grad()
	# loss_kind.backward()
	# l_kind.optimizer.step()
	# l_two_kind.optimizer.zero_grad()
	# loss_two_kind.backward()
	# l_two_kind.optimizer.step()



	running_average.append(env.points)
	loss_history.append(sum([loss1, loss2, loss3, loss4]))
	if (game%epoch_length == 0):
		points = sum(running_average)/epoch_length
		avg_loss = math.exp((sum(loss_history)/epoch_length).item())
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