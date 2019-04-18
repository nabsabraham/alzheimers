#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:43:36 2019

@author: nabila
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from data import get_dataloaders
import utils

data_path = "/home/nabila/Desktop/datasets/ADNI/Processed_32_per_subject"
val_split = 0.2
batch_size = 32
epochs = 50

train_loader, val_loader = get_dataloaders(data_path, val_split=False, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class newVGG(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).cuda()
        self.features = self.vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(nn.Linear(8192,256), 
                                   nn.ReLU(),
                                   nn.Dropout(0.5), 
                                   nn.Linear(256,n_classes), 
                                   nn.LogSoftmax(dim=1))
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
                
#images, labels = next(iter(train_loader))
#classes = ['AD','MC','Normal']
#plt.imshow(vutils.make_grid(images, nrow=4).permute(2,1,0))
#print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))    

model = newVGG(n_classes=3)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.003)

print('Training...')
for epoch in range(epochs):
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()

    else:
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()
        print("Training loss:", running_loss/len(train_loader))
  
dataiter = iter(val_loader)
images, labels = dataiter.next()
img = images[1]

ps = torch.exp(model(img))

# Plot the image and probabilities
utils.view_classify(img, ps)