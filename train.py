import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch.optim as optim


import os
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import skimage.transform

import cv2

import kNet
import dataset


net = kNet.LidarNet()
print(net)

# empty space is 70% so weight occupied space by 0.7
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1000.0]))


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

BKP_DIR = "checkpoints/"
SAVE_EVERY = 50

GAMMA = 0.0000001

# TRANSFORMS = transforms.Compose([
# 								 transforms.ToPILImage(),
# 								 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
# 								 transforms.ToTensor()
# 								 ])
# TRANSFORMS_TEST = transforms.Compose([
# 								 transforms.ToPILImage(),
# 								 transforms.ToTensor()
# 								 ])

train_dataset = dataset.LidarDataset(annotations="1.txt",
									annotation_dir="data/",
									input_dim=(4, 16, 1000),
									transform=None)
print "Loaded train_dataset..."

test_dataset = dataset.LidarDataset(annotations="1.txt",
									annotation_dir="data/",
									input_dim=(4, 16, 1000),
									transform=None)
print "Loaded test_dataset..."

BATCHSIZE = 8

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=False)

##### loading model and optimizer state #####

RESUME = True
START_EPOCH = 0
RESTORE_MODEL_PATH = BKP_DIR + "0.pth"
# if RESUME:
# 	print "Resuming training..."
# 	print "loading model: ", RESTORE_MODEL_PATH
# 	netA = torch.load(RESTORE_MODEL_PATH)

if RESUME:
	if os.path.isfile(RESTORE_MODEL_PATH):
		# print("=> loading checkpoint '{}'".format(args.resume))
		print "Resuming training..."
		print "loading model: ", RESTORE_MODEL_PATH
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
		# print "minibatch no.:", i
		# print data
		# get the inputs
		inputs, labels = data['image'], data['keypoints']
		# print inputs.shape
		# print labels

		# wrap them in Variable
		# inputs, labels = Variable(inputs), Variable(labels)
		inputs = Variable(inputs).float()
		labels = Variable(labels).long()
		# use torch.cat to concatenate tensors!

		# inputs = torch.unsqueeze(inputs, 0)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)

		# print outputs.shape, labels.shape
		outputs = outputs.view(outputs.size(0), 2, 16, -1)
		# print outputs.shape

		data_loss = criterion(outputs, labels)
		

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
				# inputs, labels = Variable(inputs), Variable(labels)
				inputs = Variable(inputs).float()
				labels = Variable(labels).long()
				# use torch.cat to concatenate tensors!

				# inputs = torch.unsqueeze(inputs, 0)

				# forward
				outputs = net(inputs)

				# print outputs.shape, labels.shape
				outputs = outputs.view(outputs.size(0), 2, 16, -1)
				# print outputs.shape

				data_loss_test = criterion(outputs, labels)
				
				test_loss += data_loss_test # + GAMMA*regularization_loss_test

				# only for prining purposes
				data_loss_test = 0.0
				
				# test_loss = test_loss + loss

				# print test_output
				# print net(torch.unsqueeze(test_input,0))

			# print('[%d, %5d] test loss: %.10f' %
			# 	  (epoch + 1, i + 1, test_loss*1.0 / len(testloader)))
			print('[%d, %5d] test loss: %.10f data_loss: %.10f reg_loss: %.10f' %
				  (epoch + 1, i + 1, test_loss*1.0 / test_sample_count, data_loss_test*1.0 / test_sample_count ,\
				  						regularization_loss_test*1.0 / test_sample_count))
			
			net.train()

	# if epoch%SAVE_EVERY == 0:
	# 	print("Saving weights...")
	# 	torch.save(net.state_dict(), BKP_DIR + str(epoch) + ".pth")

	if epoch%SAVE_EVERY == 0:
		print("Saving weights...")
		# torch.save(netA.state_dict(), BKP_DIR + str(epoch) + ".pth")
		# torch.save(netA, BKP_DIR + str(epoch) + ".pth")
		save_checkpoint({
			'epoch': epoch + 1,
			# 'arch': args.arch,
			'state_dict': net.state_dict(),
			# 'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, BKP_DIR, str(epoch) + ".pth")


print('Finished Training')