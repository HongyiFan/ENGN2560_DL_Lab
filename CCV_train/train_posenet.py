import os
import time

import torch
import torchvision

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

import PoseNet
from DataSource import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

learning_rate = 0.0001
batch_size = 75
EPOCH = 40000
directory = 'KingsCollege/'

datasource = DataSource(directory, train=True)
train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

posenet = PoseNet.posenet_v1().to(device)

posenet.cuda()

criterion = PoseNet.PoseLoss(0.3, 150, 0.3, 150, 1, 500)

optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate)

for epoch in range(EPOCH):

	t_start = time.time()

	for step, (images, poses) in enumerate(train_loader):

		b_images = Variable(images, requires_grad=True).to(device)
		poses[0] = np.array(poses[0])
		poses[1] = np.array(poses[1])
		poses[2] = np.array(poses[2])
		poses[3] = np.array(poses[3])
		poses[4] = np.array(poses[4])
		poses[5] = np.array(poses[4])
		poses[6] = np.array(poses[5])
		poses = np.transpose(poses)
		b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

		p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
		loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	elapsed_time = (time.time() - t_start)

	if epoch % 100 == 0:

		torch.save(posenet.state_dict(), 'posenet_state.chk')

		print("EPOCH " + str(epoch) + " completed in " + str(int(elapsed_time)) + " secs. Loss: " + str(loss.data.cpu().numpy()))

