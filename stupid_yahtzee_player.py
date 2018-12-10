import random
import yahtzee as yt 

env = yt.mini_environment(2)
num_games = 1000000
rounds_per_game = 2

for game in range(num_games):
	env.reset()
	while(True):
		# take an action in the environment and get a reward
		# print(state.shape)
		dice = env.dice
		save = []
		for i in range(5):
			if dice[i] > 3:
				save.append(1)
			else:
				save.append(0)


		reward = env.score(save)

		if env.rounds == 0:
			break

	
	if (game%10000 == 0):
		print("average: ", str(env.total/rounds_per_game))
