import random
import yahtzee as yt 

env = yt.mini_environment(5,6)
num_games = 1000000
rounds_per_game = 2
running_points = []

for game in range(num_games):
	env.reset()
	for i in range(2):
		# take an action in the environment and get a reward
		# print(state.shape)
		dice = env.dice
		save = ""
		for i in range(5):
			if dice[i] > 3:
				save += "1"
			else:
				save += "0"

		env.roll(save)
		env.score()
		
	running_points.append(env.points)
	
	if (game%10000 == 0):
		print("average: ", str(sum(running_points)/10000))
		running_points = []
