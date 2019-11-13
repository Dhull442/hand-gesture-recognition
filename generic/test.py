import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 7)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1296, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net.load_state_dict(torch.load("part2.m"))
cap =cv2.VideoCapture(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("starting camera")
labels = ['left','right','stop','junk']
kernel = np.ones((3,3),np.uint8)
backSub =cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=8, detectShadows=False)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
lower = np.array([108, 23, 82], dtype = "uint8")
upper = np.array([179, 255, 255], dtype = "uint8")
rr = np.array([20,80,80], dtype = "int")
while True:
    ret,frame = cap.read()
    if(ret == False):
        break;Z
    width = 50
    height = 50
    dim = (width, height)
    copy = frame
    nemo = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_nemo, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    hsv_d = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    cv2.imshow('mask',hsv_d)
    resized = cv2.resize(hsv_d, dim, interpolation =cv2.INTER_AREA)
    image = cv2.equalizeHist(resized)/255.0
    inputs = torch.from_numpy(np.asarray([[image]])).float()
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        label = labels[predicted]
        font                   =cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,200)
        fontScale              = 1
        fontColor              = (0,100,255)
        lineType               = 2

        cv2.putText(copy,label,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    cv2.imshow('frame',frame)
    out.write(copy)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        color = np.array([0,0,0])
        color[0] = int(np.mean(hsv_nemo[310:330,230:250,0]))
        color[1] = int(np.mean(hsv_nemo[310:330,230:250,1]))
        color[2] = int(np.mean(hsv_nemo[310:330,230:250,2]))
        lower[0] = max(color[0] - rr[0],0)
        lower[1] = max(color[1] - rr[1],0)
        lower[2] = max(color[2] - rr[2],0)
        upper[0] = min(color[0] + rr[0],255)
        upper[1] = min(color[1] + rr[1],255)
        upper[2] = min(color[2] + rr[2],255)
out.release()
cv2.destroyAllWindows();
cap.release();
