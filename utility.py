import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader, Dataset

def readfile(path, label):
	# label 是一個 boolean variable，代表需不需要回傳 y 值
	image_dir = sorted(os.listdir(path))
	x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
	y = np.zeros((len(image_dir)), dtype=np.uint8)
	for i, file in enumerate(image_dir):
		img = cv2.imread(os.path.join(path, file))
		x[i] = cv2.resize(img,(128, 128))
		if label:
			y[i] = int(file.split("_")[0])
	if label:
		return x, y
	else:
		return x


class ImgDataset(Dataset):
	def __init__(self, x, y=None, transform=None):
		self.x = x
		# label is required to be a LongTensor
		self.y = y
		if y is not None:
			self.y = torch.LongTensor(y)
		self.transform = transform
	def __len__(self):
		return len(self.x)
	def __getitem__(self, index):
		X = self.x[index]
		if self.transform is not None:
			X = self.transform(X)
		if self.y is not None:
			Y = self.y[index]
			return X, Y
		else:
			return X

########################## model ###########################

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		#torch.nn.MaxPool2d(kernel_size, stride, padding)
		#input 維度 [3, 128, 128]
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [64, 64, 64]

			nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [128, 32, 32]

			nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	  # [256, 16, 16]

			nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	   # [512, 8, 8]
			
			nn.Conv2d(512, 1024, 3, 1, 1), # [512, 8, 8]
			nn.BatchNorm2d(1024),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),	   # [1024, 4, 4]
		)
		self.fc = nn.Sequential(
			nn.Linear(1024*4*4, 1024*4),
			nn.ReLU(),
			nn.Dropout(0.5,True),
			nn.Linear(1024*4, 1024),
			nn.ReLU(),
			nn.Dropout(0.5, True),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 11)
		)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return self.fc(out)
	
	def __str__(self):
		return self.cnn.__str__() + self.fc.__str__()

########################## model ###########################

if __name__ == "__main__":
	model = Classifier()
	params = model.parameters()

	nParams = sum(p.numel() for p in model.parameters())
	print(nParams)
