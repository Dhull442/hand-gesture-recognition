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
net.load_state_dict(torch.load("part1.m"))
cap =cv2.VideoCapture(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("starting camera")
labels = ['left','right','stop','junk']
kernel = np.ones((3,3),np.uint8)
backSub =cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=8, detectShadows=False)
while True:
    ret,frame = cap.read()
    if(ret == False):
        break;
    width = 50
    height = 50
    dim = (width, height)
    edged=cv2.Canny(frame,50,150)
    contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,contours,-1,(0,0,0),3)
    blur = cv2.blur(frame,(5,5))
    gray_frame =cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_frame, dim, interpolation =cv2.INTER_AREA)
    image = cv2.equalizeHist(resized)/255.0
    inputs = torch.from_numpy(np.asarray([[image]])).float()
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        label = labels[predicted]
        font                   =cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,200)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(frame,label,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cv2.destroyAllWindows();
cap.release();
