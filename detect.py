import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

def create_plt(gt, input):
	print("GT shape: {}".format(gt.shape))
	print("Input shape: {}".format(input[0].shape))
	
	CLASS_ID = [0, 1]
	X, Y, Z, color = [], [], [], []
	for class_id in CLASS_ID:
		for ring_id in range(16):
			x = input[0, 0, ring_id]
			y = input[0, 1, ring_id]
			z = input[0, 2, ring_id]

			# print("X:{}".format(x[gt[ring_id] == class_id].shape))
			# print("Y:{}".format(y[gt[ring_id] == class_id].shape))
			# print("Z:{}".format(z[gt[ring_id] == class_id].shape))

			X.extend(x[gt[ring_id, :] == class_id])
			Y.extend(y[gt[ring_id, :] == class_id])
			Z.extend(z[gt[ring_id, :] == class_id])
			if class_id == 0:
				color.extend(['b']*x[gt[ring_id] == class_id].shape[0])
			else:
				color.extend(['r']*x[gt[ring_id] == class_id].shape[0])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	scale =1.0
	for i in range(len(X)):
		X[i] = scale*X[i]
		Y[i] = scale*Y[i]
		Z[i] = scale*Z[i]

	# ax.scatter(X, Y, Z, s=0.2, c=color)
	# plt.axis([-3.5, 3.5, -3.5, 3.5])
	# plt.show()

	return (X, Y, Z, color)

	

# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10000.0]))
# criterion_unweighted = nn.CrossEntropyLoss()

USE_NET = "lidar"

CLASS_WEIGHTS = [1.0, 10000.0, 0.0]
COLS = 2048
MAX_ROLL = COLS

test_dataset = dataset.LidarDataset(annotations="viz8.txt",
									annotation_dir="data/",
									input_dim=(5, 16, COLS), #1024),
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

RESTORE_MODEL_PATH = BKP_DIR + '300_euler.pth'

lidar_classify = detect(RESTORE_MODEL_PATH, USE_NET)

elapsed_time = 0.0
forward_passes = 0

output_to_plot = []

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

	print("INput shape:{}, Output shape: {}".format(img_gt.shape, outputs.data.numpy()[0].shape))
	
	output_to_plot.append(create_plt(outputs.data.numpy()[0], inputs.data.numpy()))
	# output_to_plot.append(create_plt(img_gt, inputs.data.numpy()))
	# break

	img_gt[img_gt == 2] = 0
	img_gt = np.repeat(img_gt, 10, axis=0)

	# cv2.imshow("predictions", 255*img_pred.astype(np.uint8))
	# cv2.imshow("gt", 255*img_gt.astype(np.uint8))
	# print len(np.where(img_gt == 1)[0])
	# print type(img_gt), img_gt.dtype
	# cv2.waitKey(0)







fig = plt.figure()
ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
X, Y, Z, color = [], [], [], []
sc = ax.scatter(X, Y, Z, s=10, c=color)#, c=color)
plt.axis([-4.5, 4.5, -4.5, 4.5])
# plt.show()


def animate(i):
	x, y, z, C = output_to_plot[i][0], output_to_plot[i][1], output_to_plot[i][2], output_to_plot[i][3]
	print i, len(x), len(y), len(z), len(C)

	sc._offsets3d = (x, y, z)
	sc._facecolor3d = C
	return (sc)


ani = animation.FuncAnimation(fig, animate, 
				frames=317, interval=20, repeat=True)
# test:
# viz5 125

# train:
# viz7 701
# viz8 317
# viz9 
# viz10 

plt.show()