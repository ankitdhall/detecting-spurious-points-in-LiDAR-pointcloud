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
from skimage import img_as_ubyte

import cv2
import sys

class Roll(object):
    """Roll the LiDAR scan in order to mimic a physical rotation of the LiDAR orientation

    Args: max_roll 
    """

    def __init__(self, max_roll):
        self.max_roll = max_roll

    def __call__(self, sample):
        image, output = sample['image'], sample['keypoints']

        roll_ahead = np.random.randint(0, self.max_roll)

        # print image.shape, output.shape

        image = np.roll(image, roll_ahead, axis=2)
        output = np.roll(output, roll_ahead, axis=1)

        return {'image': image, 'keypoints': output}

class LidarDataset(Dataset):
	"""Cones Landmarks dataset."""
	def __init__(self, annotations, annotation_dir, input_dim, dont_read, normalize=False, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.INPUT_DIM = input_dim

		self.annotation_dir = annotation_dir
		self.dont_read = dont_read

		text_file = open(annotations, "r")
		filename = text_file.read().strip().split("\n")
		text_file.close()

		for i in range(len(filename)):
			filename[i] = filename[i][:-4]

		self.filename = filename
		self.transform = transform

		# self.target = []
		# self.input = []
		# # annotation_dir = "/home/ankit/code/AMZ/keypoints/annotations/"
		# for i in range(len(filename)):
		# 	print filename[i]
		# 	# loaded = np.load(annotation_dir + filename[i] + ".npz")
		# 	# decoded_file = loaded["gt"]
		# 	# decoded_file = decoded_file[()]
		# 	loaded = np.load(annotation_dir + filename[i] + ".npy")
		# 	decoded_file = loaded.item()
		# 	print decoded_file.keys()
		# 	print len(decoded_file)

		# 	data_in = np.zeros(self.INPUT_DIM)
		# 	data_out = np.zeros((self.INPUT_DIM[1], self.INPUT_DIM[2]))

		# 	for ring_id in decoded_file.keys():
		# 		if ring_id in dont_read:
		# 			continue
				
				
					
		# 		ring = decoded_file[ring_id]
		# 		print ring_id, len(ring["x_gt"])
				
		# 		copy_till = min(self.INPUT_DIM[2], len(ring["y_gt"]))

		# 		data_in[0, ring_id, :copy_till] = ring["x_gt"][:copy_till]
		# 		data_in[1, ring_id, :copy_till] = ring["y_gt"][:copy_till]
		# 		data_in[2, ring_id, :copy_till] = ring["z_gt"][:copy_till]
		# 		data_in[3, ring_id, :copy_till] = ring["i_gt"][:copy_till]
				
		# 		data_out[ring_id, :copy_till] = [int(item) for item in ring["label_gt"][:copy_till]]

		# 	self.target.append(data_out)
		# 	self.input.append(data_in)

		

	def __len__(self):
		return len(self.filename)

	def __getitem__(self, idx):		

		# print self.filename[idx]
		# loaded = np.load(annotation_dir + filename[i] + ".npz")
		# decoded_file = loaded["gt"]
		# decoded_file = decoded_file[()]
		loaded = np.load(self.annotation_dir + self.filename[idx] + ".npy")
		decoded_file = loaded.item()
		# print decoded_file.keys()
		# print len(decoded_file)

		data_in = np.zeros(self.INPUT_DIM)
		data_out = np.zeros((self.INPUT_DIM[1], self.INPUT_DIM[2]))

		for ring_id in decoded_file.keys():
			if ring_id in self.dont_read:
				continue
			
			
				
			ring = decoded_file[ring_id]
			# print ring_id, len(ring["x_gt"])
			
			copy_till = min(self.INPUT_DIM[2], len(ring["y_gt"]))

			data_in[0, ring_id, :copy_till] = ring["x_gt"][:copy_till]
			data_in[1, ring_id, :copy_till] = ring["y_gt"][:copy_till]
			data_in[2, ring_id, :copy_till] = ring["z_gt"][:copy_till]
			data_in[3, ring_id, :copy_till] = ring["i_gt"][:copy_till]
			
			data_out[ring_id, :copy_till] = [int(item) for item in ring["label_gt"][:copy_till]]


		if self.transform:
			sample = self.transform({'image': data_in, 'keypoints': data_out})

		else:
			sample = {'image': data_in, 'keypoints': data_out}
		return sample


class ConesLandmarksDataset(Dataset):
	"""Cones Landmarks dataset."""

	def __init__(self, annotations, annotation_dir, img_dir, input_dim, normalize=False, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.INPUT_DIM = input_dim

		text_file = open(annotations, "r")
		filename = text_file.read().strip().split("\n")
		text_file.close()

		for i in range(len(filename)):
			filename[i] = filename[i][:-4]

		self.filename = filename

		self.target = []
		# annotation_dir = "/home/ankit/code/AMZ/keypoints/annotations/"
		for i in range(len(filename)):
			text_file = open(annotation_dir + filename[i] + ".txt", "r")
			img_annotation = text_file.read().strip().split("\n")

			# print filename[i]
			# print annotation_dir + filename[i] + ".txt"

			img_cols, img_rows, img_channels = img_annotation[0].split(" ")[0:3]
			# print img_cols, img_rows, img_channels

			dw = 1.0/int(img_cols)
			dh = 1.0/int(img_rows)
			

			annotation = []
			for ii in range(1, len(img_annotation)-1):
				# print img_annotation[i].split(" ")
				c,r = img_annotation[ii].split(" ")[0:2]

				if normalize:
					annotation.append(float(1.0*int(c)*dw))
					annotation.append(float(1.0*int(r)*dh))
				else:
					annotation.append(float(1.0*int(c)*dw*self.INPUT_DIM))
					annotation.append(float(1.0*int(r)*dh*self.INPUT_DIM))



			annotation = np.array(annotation)
			self.target.append(annotation)
			# print np.where(annotation > 1.0)


			if len(np.where(annotation > 1.0)[0]) != 0 and normalize == True:
				print img_annotation
				print annotation
				print filename[i]
				print "WARNING! annotation exceeds dimension!"
				sys.exit()
			

			text_file.close()

		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.target)

	def __getitem__(self, idx):
		img_name = self.img_dir + self.filename[idx] + ".jpg"
		resized_img = skimage.transform.resize(io.imread(img_name), (self.INPUT_DIM, self.INPUT_DIM, 3))
		image = resized_img

		# print image.shape
		image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_BGR2RGB)
		# cv2.imshow("cones BGR", image)
		# cv2.waitKey(0)

		# print image.dtype
		# print image

		# image = np.transpose(image, (2, 0, 1))

		# image = torch.from_numpy(resized_img)
		# # print image.shape # 104, 74, 3 BGR
		# image = image.permute(2, 0, 1).numpy()

		# print image.dtype
		# print "dataset", image.shape

		
		

		keypoints = torch.from_numpy(self.target[idx])
		# print keypoints

		if self.transform:
			image = self.transform(image)

		sample = {'image': image, 'keypoints': keypoints}

		

		return sample
