import math
import random
import numpy as np 
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional


# class DQN_dice(nn.Module):
# 	def __init__(self, input_channels, output_channels):
# 		super(DQN_dice, self).__init__()
# 		self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=4)
# 		self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
# 		self.head = nn.Linear(448, output_channels)

# 	def forward(self, x):
# 		x = functional.relu(self.conv1(x))
# 		x = functional.relu(self.conv2(x))
# 		prediction = self.head(x.view(x.size(0), -1))
# 		return prediction

# # for finding a yahtzee or a chance
# class DQN_toy(nn.Module):
# 	def __init__(self):
# 		super(DQN_toy, self).__init__()
# 		self.conv1 = nn.Conv1d(6, 64, kernel_size=3, stride=2)
# 		self.conv2 = nn.Conv1d(64, 364, kernel_size=4, stride=2)
# 		self.head = nn.Linear(64, 32)
# 		self.optimizer = optim.RMSprop(self.parameters())

# 	def forward(self, x):
# 		x = functional.relu(self.conv1(x))
# 		x = functional.relu(self.conv2(x))
# 		output = self.head(x.view(x.size(0), -1))
# 		return output.data[0]


# class DQN(nn.Module):
# 	def __init__(self):
# 		super(DQN, self).__init__()
# 		self.conv1 = nn.Conv1d(6, 64, kernel_size=3, stride=2)
# 		self.conv2 = nn.Conv1d(64, 64, kernel_size=4, stride=2)
# 		self.head = nn.Linear(64, 32)
# 		self.drop = nn.Dropout(p=0.6)
#         self.sm = nn.Softmax(dim=0)
#         self.action_history = Variable(torch.Tensor(), requires_grad=True)
#         self.optimizer = optim.Adam(self.parameters(), lr=.001)


# 	def forward(self, x):
# 		x = functional.relu(self.conv1(x))
# 		x = functional.relu(self.conv2(x))
# 		return self.sm(self.head(x))


class DQN(nn.Module):
	def __init__(self):
<<<<<<< HEAD
		super(DQN, self).__init__()
		self.conv1 = nn.Conv1d(5, 32, kernel_size=3, stride= 8)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm1d(32)
		self.bn2 = nn.BatchNorm1d(64)
		self.head = nn.Linear(64, 32)
		self.drop = nn.Dropout(p=0.6)
		self.sm = nn.Softmax(dim=0)
		self.action_history = Variable(torch.Tensor(), requires_grad=True)
		self.optimizer = optim.Adam(self.parameters(), lr=.001)


	def forward(self, x):
		x = functional.relu(self.bn1(self.conv1(x)))
		x = self.drop(functional.relu(self.bn2(self.conv2(x))))
		return self.sm(self.head(x))


torch.autograd.enable_grad

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.l1 = nn.Linear(in_channels, 128)
        self.l2 = nn.Linear(128, out_channels)
        self.sm = nn.Softmax(dim=0)
        self.drop = nn.Dropout(p=0.6)
        self.action_history = Variable(torch.Tensor(), requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=.001)


    def forward(self, x):
        x = self.drop(self.l1(x))
        return self.sm(self.l2(x))



class Stupid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stupid, self).__init__()
        self.l = nn.Linear(in_channels, out_channels)
        self.sm = nn.Softmax(dim=0)
        self.optimizer = optim.SGD(self.parameters(), lr=.001)


    def forward(self, x):
        return self.sm(self.l(x))