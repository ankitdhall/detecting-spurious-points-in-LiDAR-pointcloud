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


def dist(p, q):
	return ((p[0]-q[0])**2 + (p[1]-q[1])**2 )**0.5

def get_cross_ratio(keypoints):
	# print len(keypoints)
	a = (keypoints[0], keypoints[1])
	b = (keypoints[2], keypoints[3])
	c = (keypoints[4], keypoints[5])
	d = (keypoints[6], keypoints[7])
	w = a
	x = (keypoints[8], keypoints[9])
	y = (keypoints[10], keypoints[11])
	z = (keypoints[12], keypoints[13])
	return (dist(a, c)/dist(c, b))/(dist(a, d)/dist(d, b)), (dist(w, y)/dist(y, x))/(dist(w, z)/dist(z, x))

def get_reg_loss_as_lists(outputs, BATCHSIZE):
	retval_l, retval_r = [], []
	for i in range(outputs.shape[0]):
		l, r = get_cross_ratio(outputs[i])
		retval_l.append(l)
		retval_r.append(r)

	return retval_l, retval_r

EPS = 1e-5
def cross_ratio_calc(target):
	var_l = (((target[:,0]-target[:,4])**2 + (target[:,1]-target[:,5])**2 + EPS)**0.5 / ((target[:,4]-target[:,2])**2 + (target[:,5]-target[:,3])**2 + EPS)**0.5) / \
	(((target[:,0]-target[:,6])**2 + (target[:,1]-target[:,7])**2 + EPS)**0.5 / ((target[:,6]-target[:,2])**2 + (target[:,7]-target[:,3])**2 + EPS)**0.5)

	var_r = (((target[:,0]-target[:,10])**2 + (target[:,1]-target[:,11])**2 + EPS)**0.5 / ((target[:,10]-target[:,8])**2 + (target[:,11]-target[:,9])**2 + EPS)**0.5) / \
	(((target[:,0]-target[:,12])**2 + (target[:,1]-target[:,13])**2 + EPS)**0.5 / ((target[:,12]-target[:,8])**2 + (target[:,13]-target[:,9])**2 + EPS)**0.5)

	return var_l, var_r

net = kNet.LidarNet()
print(net)

# empty space is 70% so weight occupied space by 0.7
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 100.0]))


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

ANNO_DIR = "/home/vision/code/keypoints/keypoints/augmentation/augmented_anno/"
IMG_DIR = "/home/vision/code/keypoints/keypoints/augmentation/augmented_img/"

ANNO_DIR = "/home/ankit/code/AMZ/keypoints/augmentation/augmented_anno/"
IMG_DIR = "/home/ankit/code/AMZ/keypoints/augmentation/augmented_img/"

BKP_DIR = "checkpoints/"
SAVE_EVERY = 100

MEASURED_CR_CONST = 1.3940842428872968
# MEASURED_CR = Variable(torch.from_numpy(np.array(MEASURED_CR, dtype=np.float)))

GAMMA = 0.0000001
# transforms.Compose([Rescale(256),
# 					RandomCrop(224)])

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

test_dataset = dataset.LidarDataset(annotations="1.txt",
									annotation_dir="data/",
									input_dim=(4, 16, 1000),
									transform=None)

BATCHSIZE = 8

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=False)


net.train()
for epoch in range(102):  # loop over the dataset multiple times

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
				  (epoch + 1, i + 1, running_loss*1.0 / len(trainloader), running_data_loss*1.0 / len(trainloader), running_reg_loss*1.0 / len(trainloader)))
			running_loss = 0.0
			running_data_loss = 0.0
			running_reg_loss = 0.0

			# testing
			test_loss, data_loss_test, regularization_loss_test = 0.0, 0.0, 0.0
			net.eval()
			for test_i, data in enumerate(testloader, 0):
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

				# test_loss = test_loss + loss

				# print test_output
				# print net(torch.unsqueeze(test_input,0))

			# print('[%d, %5d] test loss: %.10f' %
			# 	  (epoch + 1, i + 1, test_loss*1.0 / len(testloader)))
			print('[%d, %5d] test loss: %.10f data_loss: %.10f reg_loss: %.10f' %
				  (epoch + 1, i + 1, test_loss*1.0 / len(testloader), data_loss_test*1.0 / len(testloader), regularization_loss_test*1.0 / len(testloader)))
			
			net.train()

	if epoch%SAVE_EVERY == 0:
		print("Saving weights...")
		torch.save(net.state_dict(), BKP_DIR + str(epoch) + ".pth")


print('Finished Training')