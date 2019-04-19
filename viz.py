#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:27:28 2019

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
from torch import topk

import skimage.transform
import skimage
import numpy as np
import matplotlib.pyplot as plt

from data import get_dataloaders
import utils

data_path = "/home/nabila/Desktop/datasets/ADNI/Processed_32_per_subject"
val_split = 0.2
batch_size = 32
epochs = 5
lr = 1e-6
device = 'cuda' 
classes = ['AD', 'MCI', 'Normal']

train_loader, val_loader = get_dataloaders(data_path, val_split=0.2, batch_size=16, shuffle=True)

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

model = newVGG(3).device()
model.load_state_dict(torch.load('checkpoint.pt'))
    
outputs=[]
def hook(module, input, output):    
    outputs.append(output)
       
model.features.register_forward_hook(hook)

images, labels = next(iter(val_loader))
images = images.to(device)
prediction = model(images).cpu().detach()
soft_preds, preds = torch.max(prediction,1)
#pred_probabilities = F.softmax(prediction).data.squeeze()
#topk(prediction,1)

fc_weights = model._modules.get('classifier').parameters()
for i in range(3): weights = next(iter(fc_weights))
class_idx = topk(prediction,1)[1]

#overlay = utils.getCAM(outputs[-2], weights, class_idx.int())
#feature_conv = outputs[0][1,:256,:,:].cpu().detach()

idx = 12
img = images[idx] 
gt = classes[int(labels[idx])]

utils.view_classify(img, prediction[idx], gt)


features_np = outputs[-2].cpu().detach().numpy()[idx]
weights_np = weights.cpu().detach().numpy()
top_idx = class_idx[idx].numpy()

def get_cam(features_np, weights_np, top_idx):
    nc, h, w = features_np.shape
    F = features_np.reshape((nc, h*w))
    W = weights_np[top_idx]
    
    cam = np.dot(W,F)
    cam = np.reshape(cam, (h,w))
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

cam_img = get_cam(features_np, weights_np, top_idx)
plt.imshow(images[idx].cpu().detach().permute(1,2,0).numpy())
plt.imshow(skimage.transform.resize(cam_img, img.shape[1:]), alpha=0.5, cmap='jet')

#for out in outputs:
#    print(out.shape)



#img = images[idx].unsqueeze(0)
#
#ps = torch.exp(model(img))
#gt = classes[int(labels[idx])]
#utils.view_classify(img, ps, gt)