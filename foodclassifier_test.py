import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import sys
from utility import *

test_file_path = sys.argv[1]
out_file_path = sys.argv[2]

print("Reading data")
test_x = readfile(test_file_path, False)
print("Size of Testing data = {}".format(len(test_x)))

test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
])

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

model = Classifier().cuda()
model.load_state_dict( torch.load('model/cnn') )

model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open(out_file_path, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))