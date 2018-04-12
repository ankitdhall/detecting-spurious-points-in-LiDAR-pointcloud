import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LidarNet(nn.Module):

	def __init__(self):
		super(LidarNet, self).__init__()
		padval = (1, 1)

		self.bn1 = nn.BatchNorm2d(5, affine=True)
		self.conv1 = nn.Conv2d(5, 16, 3, padding=padval)
		
		self.bn2 = nn.BatchNorm2d(16, affine=True)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=padval)

		self.bn3 = nn.BatchNorm2d(32, affine=True)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=padval)

		self.bn4 = nn.BatchNorm2d(64, affine=True)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=padval)

		# self.fc1 = nn.Linear(3072, 4096)
		# self.dropfc1 = nn.Dropout(p=0.75)

		# self.fc2 = nn.Linear(4096, 2*16000)

		self.bn4Up = nn.BatchNorm2d(128, affine=True)
		self.conv4Up = nn.ConvTranspose2d(128, 64, 3, stride=(1, 4), padding=padval)

		self.bn3Up = nn.BatchNorm2d(64, affine=True)
		self.conv3Up = nn.ConvTranspose2d(64, 32, 3, stride=(1, 4), padding=padval)

		self.bn2Up = nn.BatchNorm2d(32, affine=True)
		self.conv2Up = nn.ConvTranspose2d(32, 16, 3, stride=(1, 4), padding=padval)

		self.bn1Up = nn.BatchNorm2d(16, affine=True)
		self.conv1Up = nn.ConvTranspose2d(16, 2, 3, stride=(1, 4), padding=padval)

	def forward(self, x):
		# print x
		x = self.bn1(x)
		# print x.size()
		s1 = x.size()
		x = F.relu(self.conv1(x))
		# print x.size()
		x = F.max_pool2d(x, (1, 4))
		# print x.size()

		x = self.bn2(x)
		s2 = x.size()
		# print x.size()
		x = F.relu(self.conv2(x))
		# print x.size()
		x = F.max_pool2d(x, (1, 4))
		# print x.size()

		x = self.bn3(x)
		s3 = x.size()
		# print x.size()
		x = F.relu(self.conv3(x))
		# print x.size()
		x = F.max_pool2d(x, (1, 4))
		# print x.size()

		x = self.bn4(x)
		s4 = x.size()
		# print x.size()
		x = F.relu(self.conv4(x))
		# print x.size()
		x = F.max_pool2d(x, (1, 4))
		# print x.size()

		# print "*"*64
		

		# print s1
		# print s2
		# print s3
		# print s4

		# x = x.view(x.size(0), -1)

		# x = F.relu(self.fc1(x))
		# x = self.dropfc1(x)

		# x = self.fc2(x)

		x = self.bn4Up(x)
		x = F.relu(self.conv4Up(x, output_size=s4))
		# print x.size()


		x = self.bn3Up(x)
		x = F.relu(self.conv3Up(x, output_size=s3))
		# print x.size()


		x = self.bn2Up(x)
		# x = self.ups_1_4(x)
		x = F.relu(self.conv2Up(x, output_size=s2))
		# print x.size()

		x = self.bn1Up(x)
		# x = self.ups_1_4(x)
		x = self.conv1Up(x, output_size=s1)
		# print x.size()
		

		return x

# class kNet(nn.Module):

# 	def __init__(self):
# 		super(kNet, self).__init__()

# 		self.bn1 = nn.BatchNorm2d(3, affine=True)
# 		self.conv1 = nn.Conv2d(3, 16, 3)
		
# 		self.bn2 = nn.BatchNorm2d(16, affine=True)
# 		self.conv2 = nn.Conv2d(16, 32, 3)

# 		self.bn3 = nn.BatchNorm2d(32, affine=True)
# 		self.conv3 = nn.Conv2d(32, 64, 3)

# 		self.bn4 = nn.BatchNorm2d(64, affine=True)
# 		self.conv4 = nn.Conv2d(64, 128, 3)

# 		# self.conv4 = nn.Conv2d(64, 128, 3)
# 		# self.drop4 = nn.Dropout(p=0.5)

# 		self.fc1 = nn.Linear(1152, 512)
# 		self.dropfc1 = nn.Dropout(p=0.5)

# 		self.fc2 = nn.Linear(512, 14)

# 	def forward(self, x):
# 		# print x
# 		x = self.bn1(x)
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		
# 		x = self.bn2(x)
# 		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

# 		x = self.bn3(x)
# 		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

# 		x = self.bn4(x)
# 		x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

# 		x = x.view(x.size(0), -1)

# 		x = F.relu(self.fc1(x))
# 		x = self.dropfc1(x)

# 		x = self.fc2(x)
# 		return x

# class kNetAutoencoder(nn.Module):

# 	def __init__(self):
# 		super(kNetAutoencoder, self).__init__()

# 		self.bn1 = nn.BatchNorm2d(3, affine=True)
# 		self.conv1 = nn.Conv2d(3, 16, 3)
		
# 		self.bn2 = nn.BatchNorm2d(16, affine=True)
# 		self.conv2 = nn.Conv2d(16, 32, 3)

# 		self.bn3 = nn.BatchNorm2d(32, affine=True)
# 		self.conv3 = nn.Conv2d(32, 64, 3)

# 		self.bn4 = nn.BatchNorm2d(64, affine=True)
# 		self.conv4 = nn.Conv2d(64, 128, 3)

# 		# self.fc1 = nn.Linear(1152, 512)
# 		# self.dropfc1 = nn.Dropout(p=0.5)

# 		# self.fc2 = nn.Linear(512, 14)

# 		# self.fc1Up = nn.Linear(512, 1152)
# 		# self.dropfc1Up = nn.Dropout(p=0.5)

# 		self.bn4Up = nn.BatchNorm2d(128, affine=True)
# 		self.conv4Up = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)

# 		self.bn3Up = nn.BatchNorm2d(64, affine=True)
# 		self.conv3Up = nn.ConvTranspose2d(64, 32, 3, stride=3, padding=0)

# 		self.bn2Up = nn.BatchNorm2d(32, affine=True)
# 		self.conv2Up = nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1)

# 		self.bn1Up = nn.BatchNorm2d(16, affine=True)
# 		self.conv1Up = nn.ConvTranspose2d(16, 3, 2, stride=2, padding=3)
		
		

# 	def forward(self, x):
# 		# print x
# 		x = self.bn1(x)
# 		# print x.size()
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# 		# print x.size()

# 		x = self.bn2(x)
# 		# print x.size()
# 		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
# 		# print x.size()
		
# 		x = self.bn3(x)
# 		# print x.size()
# 		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
# 		# print x.size()

# 		x = self.bn4(x)
# 		# print x.size()
# 		x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
# 		# print x.size()

# 		# x = x.view(x.size(0), -1)

# 		# x = F.relu(self.fc1(x))
# 		# x = self.dropfc1(x)

# 		# x = self.fc2(x)


# 		# x = self.fc1Up(x)

# 		x = self.bn4Up(x)
# 		x = F.relu(self.conv4Up(x))
# 		# print x.size()


# 		x = self.bn3Up(x)
# 		x = F.relu(self.conv3Up(x))
# 		# print x.size()


# 		x = self.bn2Up(x)
# 		x = F.relu(self.conv2Up(x))
# 		# print x.size()

# 		x = self.bn1Up(x)
# 		x = self.conv1Up(x)
# 		# print x.size()

# 		return x

# class kNet_old(nn.Module):

# 	def __init__(self):
# 		super(kNet, self).__init__()

# 		self.bn1 = nn.BatchNorm2d(3, affine=False)
# 		self.conv1 = nn.Conv2d(3, 16, 3)
		
# 		self.conv2 = nn.Conv2d(16, 64, 3)

# 		self.bn3 = nn.BatchNorm2d(64, affine=False)
# 		self.conv3 = nn.Conv2d(64, 128, 3)

# 		self.conv4 = nn.Conv2d(128, 64, 1)
# 		self.drop4 = nn.Dropout(p=0.5)

# 		self.conv5 = nn.Conv2d(64, 14, 8)

# 	def forward(self, x):
# 		# print x
# 		x = self.bn1(x)
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		
# 		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

# 		x = self.bn3(x)
# 		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

# 		x = F.relu(self.conv4(x))
# 		x = self.drop4(x)

# 		x = F.relu(self.conv5(x))
# 		x = x.view(-1, 14)
# 		return x

# net = kNet()
# print(net)

# netA = kNetAutoencoder()
# print(netA)


