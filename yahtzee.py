import math
import random
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim

# conversion from list to score board location
get_category = {
	0: "ones", 1: "twos", 2: "threes", 3: "fours", 4: "fives", 5: "sixes",
	6: "three_kind", 7: "four_kind", 8: "sm_straight", 9: "lg_straight",
	10: "full_house", 11: "chance", 12: "yahtzee"}

# represents a score board for yahtzee 
class environment(object):
	
	def __init__(self):
		self.total_score = 0
		self.upper_score = 0
		self.upper_bonus = False
		self.score_board = {}
		self.dirty = np.zeros((13))

		self.scoring = {
			"ones":   self.score_ones,
			"twos":   self.score_twos,
			"threes": self.score_threes,
			"fours":  self.score_fours,
			"fives":  self.score_fives,
			"sixes":  self.score_sixes,
			"three_kind": self.score_three_kind,
			"four_kind" : self.score_four_kind,
			"chance" : self.score_chance,
			"yahtzee": self.score_yahtzee
			}

	def first_roll(self):
		dice = []
		for i in range(5):
			dice.append(random.randint(1,6))

		return dice

	def get_dice(self, save, dice):
		for i in range(5):
			if(save[i] == 0):
				dice[i] = random.randint(1,6)

		return dice


	def validate_kind(self, n, dice):
		for k in range(1,7):
			if (np.where(dice == k)[0].size >= n):
				return True, k

		return False, 0

	def score_upper(self, k, dice):
		return k * np.where(dice == k)[0].size

	def score_ones(self, dice):
		return self.score_upper(1, dice)

	def score_twos(self, dice):
		return self.score_upper(2, dice)

	def score_threes(self, dice):
		return self.score_upper(3, dice)

	def score_fours(self, dice):
		return self.score_upper(4, dice)

	def score_fives(self, dice):
		return self.score_upper(5, dice)

	def score_sixes(self, dice):
		return self.score_upper(6, dice)


	def score_three_kind(self, dice):
		validation, k = self.validate_kind(3, dice)
		return k * 3

	def score_four_kind(self, dice):
		validation, k = self.validate_kind(4, dice)
		return k * 4

	def score_chance(self, dice):
		return np.sum(dice)

	def score_yahtzee(self, dice):
		validation, k = self.validate_kind(5, dice)
		if validation:
			return 50
		else:
			return 0


	# category is the string representation of the scoreboard slot, and dice
	# is a length 5 vector of dice amounts. returns dice score
	def step(self, action, dice):
		category = get_category[action]
		if category in self.score_board:
			if (category == "yahtzee") and self.validate_kind(5, dice):
				if self.score_board[category] != 0:
					self.score_board[category] += 50
					return 50, self.dirty, False
			return 0, None, False

		score = self.scoring.get(category)
		s = score(dice)

		self.score_board[category] = s
		self.dirty[action] = 1
		gameover = np.sum(self.dirty) == 13

		return (s, self.dirty, gameover)


class mini_environment(object):
	def __init__(self):
		self.rounds = 0
		self.total = 0
		self.dice = []
		self.points = 0
		self.reset_dice()
		self.score()

	def reset_dice(self):
		dice = []
		for i in range(5):
			dice.append(random.randint(1,6))

		self.dice = dice

	def roll_dice(self, save):
		newdice = []
		mask = '{0:05b}'.format(save)
		lookup = {}
		for i in range(5):
			lookup[i] = (mask[i] == '1')
		for i in range(5):
			if (lookup[i]):
				newdice.append(self.dice[i])
			else: 
				newdice.append(random.randint(1,6))
		newdice.sort()
		self.dice = newdice

	def validate_yahtzee(self):
		arr = np.asarray(self.dice)
		for k in range(1,7):
			test = arr[np.where(arr==k)]
			if (len(test) == 5):
				return True

		return False

	def score_chance(self):
		return np.sum(self.dice)


	def score(self):
		if self.validate_yahtzee():
			self.points = 50
		else:
			self.points = self.score_chance()


	def step(self, save):
		self.roll_dice(save)
		self.score()
		self.rounds += 1
			

	def reset(self):
		self.rounds = 0
		self.reset_dice()
		self.score()

# env = mini_environment()
# for i in range(32):
# 	env.dice = [0,0,0,0,0]
# 	env.step(i)
# 	print(env.dice)