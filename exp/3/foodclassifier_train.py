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

torch.manual_seed(0)

train_file_path = os.path.join(sys.argv[1], 'training')
valid_file_path = os.path.join(sys.argv[1], 'validation')

######################## read data #########################

#分別將 training set、validation set 用 readfile 函式讀進來
print("Reading data")

train_x, train_y = readfile(train_file_path, True)
print("Size of training data = {}".format(len(train_x)))

val_x, val_y = readfile(valid_file_path, True)
print("Size of validation data = {}".format(len(val_x)))

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)

mean_std = [np.mean(train_val_x, axis=(0, 1, 2))/255, np.std(train_val_x, axis=(0, 1, 2))/255]
np.save('model/mean_std.npy', mean_std)

######################## read data #########################



####################### prepare data #######################

#training 時做 data augmentation
train_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
	# transforms.RandomRotation(15), #隨機旋轉圖片
	transforms.RandomAffine(30, (0.1, 0.1), (0.9,1.1)),
	transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
	transforms.Normalize(mean_std[0], mean_std[1]),
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean_std[0], mean_std[1]),
])

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

####################### prepare data #######################



########################## train ###########################

model = Classifier().cuda()
print(model)
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 100

log_train_acc = []
log_val_acc = []


for epoch in range(num_epoch):
	epoch_start_time = time.time()
	train_acc = 0.0
	train_loss = 0.0
	val_acc = 0.0
	val_loss = 0.0

	model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
	for i, data in enumerate(train_loader):
		optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
		train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
		batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
		batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
		optimizer.step() # 以 optimizer 用 gradient 更新參數值

		train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
		train_loss += batch_loss.item()

	log_train_acc.append(train_acc/train_set.__len__())
	
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			val_pred = model(data[0].cuda())
			batch_loss = loss(val_pred, data[1].cuda())

			val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
			val_loss += batch_loss.item()

		log_val_acc.append(val_acc/val_set.__len__())
		#將結果 print 出來
		print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
			(epoch + 1, num_epoch, time.time()-epoch_start_time, \
			 train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))


train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 100

for epoch in range(num_epoch):
	epoch_start_time = time.time()
	train_acc = 0.0
	train_loss = 0.0

	model_best.train()
	for i, data in enumerate(train_val_loader):
		optimizer.zero_grad()
		train_pred = model_best(data[0].cuda())
		batch_loss = loss(train_pred, data[1].cuda())
		batch_loss.backward()
		optimizer.step()

		train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
		train_loss += batch_loss.item()

	log_train_acc.append(train_acc/train_val_set.__len__())
	print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
	(epoch + 1, num_epoch, time.time()-epoch_start_time, \
	train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

print('\n')

torch.save(model_best.state_dict(), 'model/cnn')

np.save('log/log_train_acc.npy', log_train_acc)
np.save('log/log_val_acc.npy', log_val_acc)

########################## train ###########################