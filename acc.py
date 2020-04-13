import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

import sys
from utility import *

train_file_path = os.path.join(sys.argv[1], 'training')
valid_file_path = os.path.join(sys.argv[1], 'validation')

print("Reading data")
train_x, train_y = readfile(train_file_path, True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(valid_file_path, True)
print("Size of validation data = {}".format(len(val_x)))

mean_std = np.load('model/mean_std.npy')

test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean_std[0], mean_std[1]),
])

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, test_transform)

train_val_loader = DataLoader(train_val_set, batch_size=128, shuffle=True)

model = Classifier().cuda()
model.load_state_dict( torch.load('model/cnn') )

train_acc = 0

with torch.no_grad():
	for i, data in enumerate(train_val_loader):
		train_pred = model(data[0].cuda())
		train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

print( 'Train Acc: ', train_acc / len(train_val_set) )