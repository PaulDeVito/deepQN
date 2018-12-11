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
	def __init__(self, num_dice, dice_range):
		self.rounds = 0
		self.dice = []
		self.points = 0
		self.num_dice = num_dice
		self.dice_range = dice_range
		self.num_max_dice_rerolled = 0
		self.reset_dice()
		self.score()

	def reset_dice(self):
		dice = []
		for i in range(self.num_dice):
			dice.append(random.randint(1,self.dice_range))

		# dice.sort()
		self.dice = dice

	def get_mask(self):
		if self.num_dice == 5:
			return '{0:05b}'
		elif self.num_dice == 3:
			return '{0:03b}'
		elif self.num_dice == 2:
			return '{0:02b}'

	def roll_dice(self, save):
		# print("old dice: ", self.dice)
		# print(save)
		newdice = []
		mask = self.get_mask().format(save)
		# print(mask)
		lookup = {}
		for i in range(self.num_dice):
			lookup[i] = (mask[i] == '1')
		# print(lookup)
		for i in range(self.num_dice):
			if (lookup[i]):
				newdice.append(self.dice[i])
			else: 
				newdice.append(random.randint(1,self.dice_range))
				if (self.dice[i] == self.dice_range):
					self.num_max_dice_rerolled += 1
		# print("new dice: ", newdice)
		# newdice.sort()
		self.dice = newdice


	def validate_yahtzee(self):
		arr = np.asarray(self.dice)
		for k in range(1,7):
			test = arr[np.where(arr==k)]
			if (len(test) == 5):
				return True

		return False

	def score_chance(self):
		return sum(self.dice)


	def score(self):
		if self.validate_yahtzee():
			self.points = 50
		else:
			self.points = self.score_chance()

	def step_simple(self, save):
		self.roll_dice(save)
		self.points = self.score_chance()
		self.rounds += 1

	def step(self, save):
		self.roll_dice(save)
		self.score()
		self.rounds += 1
			

	def reset(self):
		self.rounds = 0
		self.num_max_dice_rerolled = 0
		self.reset_dice()
		self.score()


class full_environment(object):
	def __init__(self):
		self.rounds = 0
		self.total = 0
		self.dice = []
		self.points = 0
		self.scorable = {
			0: True,
			1: False,
			2: True,
			3: False,
			4: False,
			5: False,
			6: False,
			7: True,
			8: False
		}

		self.reset_dice()
		self.score()

	def reset_dice(self):
		dice = []
		for i in range(5):
			dice.append(random.randint(1,6))

		self.dice = sort(dice)

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

	def is_flush(self, frequency):
		odds = frequency[1] + frequency[3] + frequency[5]
		return ((odds == 5) or (odds == 0))

	def is_small_straight(self, frequency):
		if (frequency[3] >= 1) and (frequency[4] >= 1):
			a = (frequency[1] >= 1) and (frequency[2] >= 1)
			b = (frequency[2] >= 1) and (frequency[5] >= 1)
			c = (frequency[5] >= 1) and (frequency[6] >= 1)
			return a or (b or c)
		return False


	def score_dice(self):
		counts = {1:0, 2:0, 3:0, 4:0, 5:0}
		frequency = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

		arr = np.asarray(self.dice)
		for k in range(1,7):
			test = arr[np.where(arr==k)]
			n = len(test)
			if(n > 0):
				frequency[k] = n
				counts[n] += 1

		# check for yahtzee
		if counts[5] >= 1 and self.scorable[1]:
			return 50
		# check for large straight
		if counts[1] >= 5 and self.scorable[2]:
			if (frequency[1] == 0) or (frequency[6] == 0):
				return 50
		# check for 4 of a kind
		if counts[4] >= 1 and self.scorable[3]:
			return 40
		# check for full house
		if (counts[3] >= 1) and (counts[2] >= 1) and self.scorable[4]:
			return 30
		# check for flush
		if self.is_flush(frequency) and self.scorable[5]:
			return 25
		# small straight
		if counts[1] >= 4 and self.scorable[6]:
			if self.is_small_straight(frequency):
				return 20
		# 3 of a kind
		if counts[3] >= 1 and self.scorable[7]:
			return 10
		# two pair
		if counts[2] >= 2 and self.scorable[8]:
			return 5
		
		return 0

	def score_chance(self):
		return np.sum(self.dice)


	def score(self):
		self.points = self.score_dice()


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