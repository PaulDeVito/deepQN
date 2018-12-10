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


def optimize_model(memory, model):
	batch = Transition(*zip(*memory))

	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# print(state_batch) 
	# print(action_batch)

	associated_action = model(state_batch)
	# print(associated_action.shape)
	# print(action_batch.shape)
	# print(associated_action)
	# print(action_batch)

	# optimization computations
	# this just gets the current values of the stuff we chose before
	# state_action_values = model(state_batch).gather(1, action_batch.long())
	# so my version will be:
	# print(model(state_batch))
	# state_actions = model(state_batch) * action_batch
	# print(state_actions)
	# print(reward_batch)

	next_state_values = torch.zeros(len(memory))
	optimal_reward = (torch.ones(len(memory)) * 50)

	# optimal_long_term = 
	# print(optimal_reward)


	loss = functional.smooth_l1_loss(reward_batch, optimal_reward)
	# print(loss)
	# print(loss)

	loss = Variable(loss, requires_grad=True)

	# just three more lines need to go here

	# loss = functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	model.optimizer.zero_grad()
	loss.backward()
	for param in model.parameters():
		print(param.requires_grad)
		param.grad.data.clamp_(-1,1)
	model.optimizer.step()



num_dice = 5
num_actions = 13

dqn_dice = dqn.DQN_dice(num_dice + num_actions, num_dice)
dqn_toy = dqn.DQN_toy()
dqn_toy.train()
logistic_regression = dqn.LogisticRegression(5,5)
memory = []

rounds_per_game = 2
num_games = 1000000
env = yt.mini_environment(rounds_per_game)


i = 0
for game in range(num_games):
	env.reset()
	memory = []
	state = torch.tensor(generate_input(env.dice, env.rolls_left)).float().unsqueeze(0)
	while(True):
		# take an action in the environment and get a reward
		# print(state.shape)
		save = functional.relu(dqn_toy(state))

		# print(save)
		save[save!=0] = 1
		# print(save)
		reward = env.score(save.tolist())

		# get new state and save the transition to memory
		prev_state = state
		state = torch.tensor(generate_input(env.dice, env.rolls_left)).float().unsqueeze(0)
		if(env.rolls_left == 2):
			memory.append(Transition(prev_state, save.unsqueeze(0), state, torch.tensor([float(reward)])))
			optimize_model(memory,dqn_toy)


		if env.rounds == 0:
			break

	i += 1
	if (i%10000 == 0):
		print("average: ", str(env.total/rounds_per_game))

		#second roll
		# roll = env.get_dice(save, roll)
		# print("Second roll: ", roll)
		# inputs = torch.tensor(dice_to_input(roll)).float()
	
		# save = dqn_toy(inputs)
		# print(save)
	
		# roll = env.get_dice(save, roll)
		# print("Third roll: ", roll)