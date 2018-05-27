import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np

import sys
sys.path.append('/home/ankit/code/github/LiDAR-weather-gt')


import kNet
# import resnet
import dataset

from skimage import img_as_ubyte

import time

#ROS stuff
# to create the custom point cloud
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point

from std_msgs.msg import Header




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

			print("X:{}".format(x[gt[ring_id] == class_id].shape))
			print("Y:{}".format(y[gt[ring_id] == class_id].shape))
			print("Z:{}".format(z[gt[ring_id] == class_id].shape))

			X.extend(x[gt[ring_id, :] == class_id])
			Y.extend(y[gt[ring_id, :] == class_id])
			Z.extend(z[gt[ring_id, :] == class_id])
			if class_id == 0:
				color.extend(['b']*x[gt[ring_id] == class_id].shape[0])
			else:
				color.extend(['r']*x[gt[ring_id] == class_id].shape[0])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X, Y, Z, s=0.05, c=color)
	
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')

	plt.show()


def create_pc(gt, input):
	print("GT shape: {}".format(gt.shape))
	print("Input shape: {}".format(input[0].shape))
	
	CLASS_ID = [0, 1]
	X, Y, Z, color = [], [], [], []
	for class_id in CLASS_ID:
		for ring_id in range(16):
			x = input[0, 0, ring_id]
			y = input[0, 1, ring_id]
			z = input[0, 2, ring_id]

			print("X:{}".format(x[gt[ring_id] == class_id].shape))
			print("Y:{}".format(y[gt[ring_id] == class_id].shape))
			print("Z:{}".format(z[gt[ring_id] == class_id].shape))

			X.extend(x[gt[ring_id, :] == class_id])
			Y.extend(y[gt[ring_id, :] == class_id])
			Z.extend(z[gt[ring_id, :] == class_id])
			if class_id == 0:
				color.extend(['b']*x[gt[ring_id] == class_id].shape[0])
			else:
				color.extend(['r']*x[gt[ring_id] == class_id].shape[0])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(X, Y, Z, s=0.05, c=color)
	
	pc = []
	for ii in range(X):
		if color == 'r':
			class_id = 2
		else:
			class_id = 1
		pc.append([X[ii], Y[ii], Z[ii], class_id])

	return pc

class PublishClassifiedPC:
	def __init__(self):

		self.custom_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),

            PointField(name='class', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

		USE_NET = "lidar"

		CLASS_WEIGHTS = [1.0, 10000.0, 0.0]
		COLS = 1024
		MAX_ROLL = COLS

		test_dataset = dataset.LidarDataset(annotations="/home/ankit/code/github/LiDAR-weather-gt/test_small.txt",
											annotation_dir="/home/ankit/code/github/LiDAR-weather-gt/data/",
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

		BKP_DIR = "/home/ankit/code/github/LiDAR-weather-gt/checkpoints/"

		RESTORE_MODEL_PATH = BKP_DIR + '300.pth'

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

			pc_out = create_plt(img_gt, inputs.data.numpy())

			self.point_cloud_to_publish.append(pc_out)
			break

			# img_gt[img_gt == 2] = 0
			# img_gt = np.repeat(img_gt, 10, axis=0)

			# cv2.imshow("predictions", 255*img_pred.astype(np.uint8))
			# cv2.imshow("gt", 255*img_gt.astype(np.uint8))
			# print len(np.where(img_gt == 1)[0])
			# print type(img_gt), img_gt.dtype
			# cv2.waitKey(0)

	self.cone_pointcloud_publisher = rospy.Publisher('/points_classified', PointCloud2, queue_size=1)
	
	rospy.init_node('pc_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        point_cloud_to_publish = pc2.create_cloud(Header(stamp=rospy.Time.now(), frame_id="/velodyne"),
        											self.custom_fields,
        											self.point_cloud_to_publish[0])
        rate.sleep()