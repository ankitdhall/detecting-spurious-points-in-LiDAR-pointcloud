import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch.optim as optim


import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2

import kNet
# import resnet
import dataset

from unet_model import UNet

CLASS_WEIGHTS = [1.0, 1000.0]

# net = UNet(4, 2)
# lr = 0.01
# momentum = 0.99

net = kNet.LidarNet()
# lr = 0.01
# momentum = 0.99

# net = resnet.resnet18(pretrained=True, num_classes=2*16000)

print(net)

# empty space is 70% so weight occupied space by 0.7
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10000.0]))


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

BKP_DIR = "checkpoints/"
SAVE_EVERY = 200

GAMMA = 0.0000001


ON_CLUSTER = False
COLS = 256
INPUT_DIM = (5, 16, 1024)
MAX_ROLL = COLS

LIST_PATH = ""
ANNO_PATH = ""
if ON_CLUSTER:
	ANNO_PATH = "/cluster/scratch/adhall/"
	LIST_PATH = "/cluster/home/adhall/code/LiDAR-weather-gt/"
	BKP_DIR = LIST_PATH + BKP_DIR

train_dataset = dataset.LidarDataset(annotations=LIST_PATH + "train.txt",
									annotation_dir=ANNO_PATH + "data/",
									input_dim=INPUT_DIM,
									cols=COLS,
									dont_read=[],
									transform=dataset.Roll(MAX_ROLL))
print("Loaded train_dataset...")

test_dataset = dataset.LidarDataset(annotations=LIST_PATH + "test.txt",
									annotation_dir=ANNO_PATH + "data/",
									input_dim=INPUT_DIM,
									cols=COLS,
									dont_read=[],
									transform=dataset.Roll(MAX_ROLL))
print("Loaded test_dataset...")

BATCHSIZE = 4

# @profile
trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=True,
                                           num_workers=0)

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=False,
                                           num_workers=0)

##### loading model and optimizer state #####

RESUME = False
START_EPOCH = 0
RESTORE_MODEL_PATH = BKP_DIR + "10.pth"

if RESUME:
	if os.path.isfile(RESTORE_MODEL_PATH):
		print("Resuming training...")
		print("loading model: {}".format(RESTORE_MODEL_PATH))
		checkpoint = torch.load(RESTORE_MODEL_PATH)
		START_EPOCH = checkpoint['epoch']
		# best_prec1 = checkpoint['best_prec1']
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})"
			  .format(RESTORE_MODEL_PATH, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))

def save_checkpoint(state, dir_path, filename):
	torch.save(state, dir_path + filename)


##### loading model and optimizer state #####



net.train()
for epoch in range(START_EPOCH, 502):  # loop over the dataset multiple times

	running_loss = 0.0
	running_data_loss = 0.0
	running_reg_loss = 0.0
	# what does enumerate do?
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data['image'], data['keypoints']

		# print(inputs.shape)

		# wrap them in Variable
		inputs = Variable(inputs).float()
		labels = Variable(labels).long()
		# use torch.cat to concatenate tensors!

		# inputs = torch.unsqueeze(inputs, 0)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)

		outputs = outputs.view(outputs.size(0), 2, 16, -1)

		# data_loss = criterion(outputs, labels)
		data_loss = dataset.cross_entropy_weighted_loss(outputs, labels, CLASS_WEIGHTS)
		

		loss = data_loss # + GAMMA*regularization_loss

		loss.backward()
		optimizer.step()

		# print statistics
		running_data_loss += data_loss.data[0]
		# running_reg_loss += regularization_loss.data[0]
		running_loss += loss.data[0]

		if i % 2000 == 0:    # print every 2000 mini-batches
			print('[%d, %5d] train loss: %.10f data_loss: %.10f reg_loss: %.10f' %
				  (epoch + 1, i + 1, running_loss*1.0 / 2000.0, running_data_loss*1.0 / 2000.0, running_reg_loss*1.0 / 2000.0))
			running_loss = 0.0
			running_data_loss = 0.0
			running_reg_loss = 0.0

			# testing
			test_loss, data_loss_test, regularization_loss_test = 0.0, 0.0, 0.0
			net.eval()
			test_sample_count = 0
			for test_i, data in enumerate(testloader, 0):
				test_sample_count += 1
				inputs, labels = data['image'], data['keypoints']

				# wrap them in Variable
				inputs = Variable(inputs).float()
				labels = Variable(labels).long()

				# forward
				outputs = net(inputs)

				outputs = outputs.view(outputs.size(0), 2, 16, -1)
				
				# print torch.max(outputs, 1)


				# data_loss_test = criterion(outputs, labels)
				data_loss_test = dataset.cross_entropy_weighted_loss(outputs, labels, [1.0, 1.0])
				
				test_loss += data_loss_test # + GAMMA*regularization_loss_test

				# only for prining purposes
				data_loss_test = 0.0

			print('[%d, %5d] test loss: %.10f data_loss: %.10f reg_loss: %.10f' %
				  (epoch + 1, i + 1, test_loss*1.0 / test_sample_count, data_loss_test*1.0 / test_sample_count ,\
				  						regularization_loss_test*1.0 / test_sample_count))
			
			net.train()


	if epoch%SAVE_EVERY == 0:
		print("Saving weights...")
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': net.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, BKP_DIR, str(epoch) + ".pth")


print('Finished Training')
