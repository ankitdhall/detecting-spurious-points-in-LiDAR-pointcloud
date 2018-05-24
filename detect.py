import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np

import kNet
# import resnet
import dataset

from skimage import img_as_ubyte

import time

# class detect():

# 	def __init__(self, WEIGHTS):
# 		self.net = kNet.kNet()
# 		self.net.load_state_dict(torch.load(WEIGHTS))
# 		self.net.eval()
# 		self.criterion = nn.MSELoss()

# 	def predict(self, input):
# 		predictions = self.net(input)
# 		return predictions

# def convert_image_np(inp):
# 	"""Convert a Tensor to numpy image."""
# 	# print inp.shape
# 	inp = inp.numpy().transpose((1, 2, 0))
# 	# print inp.shape
# 	b, g, r = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]
# 	# print np.array([r, g, b]).shape
# 	return np.array([r, g, b]).transpose((1, 2, 0))

class detect():
	def __init__(self, WEIGHTS, USE_NET):
		if USE_NET == "lidar":
			self.net = kNet.LidarNet()

		checkpoint = torch.load(WEIGHTS)
		self.net.load_state_dict(checkpoint['state_dict'])
		self.net.eval()
		

	def predict(self, input):
		predictions = self.net(input)
		return predictions

# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10000.0]))
# criterion_unweighted = nn.CrossEntropyLoss()

USE_NET = "lidar"

CLASS_WEIGHTS = [1.0, 10000.0, 0.0]
COLS = 1024
MAX_ROLL = COLS

test_dataset = dataset.LidarDataset(annotations="test_small.txt",
									annotation_dir="data/",
									input_dim=(5, 16, 1024),
									cols=COLS,
									dont_read=[],
									transform=dataset.Roll(MAX_ROLL))
print("Loaded test_dataset...")

BATCHSIZE = 1

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCHSIZE, 
                                           shuffle=False,
                                           num_workers=0)

##### loading model and optimizer state #####

BKP_DIR = "checkpoints/"

RESTORE_MODEL_PATH = BKP_DIR + '60.pth'

lidar_classify = detect(RESTORE_MODEL_PATH, USE_NET)

elapsed_time = 0.0
forward_passes = 0

for test_i, data in enumerate(testloader, 0):
	inputs, labels = data['image'], data['keypoints']

	print(inputs.shape)

	print(labels)
	
	# # .copy() bug when drawing over img
	# img = np.squeeze(inputs.permute(0, 2, 3, 1).numpy()).copy()
	# # img = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2RGB)

	# wrap them in Variable
	inputs = Variable(inputs).float()
	labels = Variable(labels).long()
	# use torch.cat to concatenate tensors!

	# input_tensor = inputs.cpu().data
	# transformed_input_tensor = detect_cones.stn(inputs).cpu().data

	# in_grid = convert_image_np(
	# 	torchvision.utils.make_grid(input_tensor))

	# out_grid = convert_image_np(
	# 	torchvision.utils.make_grid(transformed_input_tensor))

	# # Plot the results side-by-side
	# f, axarr = plt.subplots(1, 2)
	# axarr[0].imshow(in_grid)
	# axarr[0].set_title('Dataset Images')

	# axarr[1].imshow(out_grid)
	# axarr[1].set_title('Transformed Images')

	# plt.show()


	t0 = time.time()

	predictions = lidar_classify.predict(inputs)

	elapsed_time += (time.time() - t0)
	forward_passes += 1

	outputs = predictions.view(predictions.size(0), len(CLASS_WEIGHTS), 16, -1)

	custom_criterion = dataset.cross_entropy_weighted_loss(outputs, labels, [1.0, 1.0, 0.0])
	custom_criterion_weighted = dataset.cross_entropy_weighted_loss(outputs, labels, CLASS_WEIGHTS)

	# print("Weigted loss:{} \nUnweighted loss:{}".format(criterion(outputs, labels), criterion_unweighted(outputs, labels)))
	print("Custom unweighted loss:{}\n Custom weighted loss:{}".format(custom_criterion, custom_criterion_weighted))

	_, outputs = torch.max(outputs, 1)
	
	print("Output:")
	print(outputs)

	print("GT:")
	print(labels)

	print("average FPS:{}".format(1.0*forward_passes/elapsed_time))

	img_pred = outputs.data.numpy()[0]
	img_pred = np.repeat(img_pred, 10, axis=0)

	img_gt = labels.data.numpy()[0]
	img_gt[img_gt == 2] = 0
	img_gt = np.repeat(img_gt, 10, axis=0)

	cv2.imshow("predictions", 255*img_pred.astype(np.uint8))
	cv2.imshow("gt", 255*img_gt.astype(np.uint8))
	print len(np.where(img_gt == 1)[0])
	print type(img_gt), img_gt.dtype
	cv2.waitKey(0)