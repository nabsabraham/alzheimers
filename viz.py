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
from torchsummary import summary
from torch import topk

import skimage.transform
import skimage
import numpy as np
import matplotlib.pyplot as plt

from data import get_dataloaders
import utils
import models

data_path = "/home/nabila/Desktop/datasets/ADNI/Processed_32_per_subject"
val_split = 0.2
batch_size = 32
epochs = 300
lr = 1e-5
device = 'cuda' 
classes = ['AD', 'MCI', 'Normal']

train_loader, val_loader = get_dataloaders(data_path, 
                                           val_split=val_split, 
                                           batch_size=batch_size, 
                                           shuffle=True)

#model = models.newVGG(3).to(device)
model = models.VGG512(3).to(device)
model.load_state_dict(torch.load('checkpoint.pt'))
    
outputs=[]
def hook(module, input, output):    
    outputs.append(output)
       
#model.features[16].register_forward_hook(hook)
model.features.register_forward_hook(hook)

images, labels = next(iter(val_loader))
images = images.to(device)
prediction = model(images).cpu().detach()
ps = torch.exp(prediction)
#soft_preds, preds = torch.max(prediction,1)
#pred_probabilities = F.softmax(prediction).data.squeeze()
#topk(prediction,1)

# get the weights and class idx for cam
fc_weights = model._modules.get('classifier').parameters()
# stupid but, the weights I need are the third iteration of the fc_weights
for i in range(3): weights = next(iter(fc_weights))
class_idx = topk(prediction,1)[1]

idx = np.random.randint(batch_size)
img = images[idx] 
gt_idx = int(labels[idx])
gt_class = classes[gt_idx]
pred_class = classes[torch.max(ps[idx],0)[1]]
features_np = outputs[-1].cpu().detach().numpy()[idx]
weights_np = weights.cpu().detach().numpy()
top_idx = class_idx[idx].numpy()

cam_img = utils.get_cam(features_np, weights_np, top_idx)
plt.figure()
plt.imshow(images[idx].cpu().detach().permute(1,2,0).numpy())
plt.imshow(skimage.transform.resize(cam_img, img.shape[1:]), alpha=0.5, cmap='jet')
plt.title(f'Predicted: {pred_class ({})} | GT: {gt_class}')
plt.axis('off')

utils.view_classify(img, ps[idx], gt_class)

#classes = ['AD', 'MCI', 'Normal']
#index = np.arange(len(classes))
#fig, (ax1, ax2) = plt.subplots(figsize=(5,7), nrows=2)
#ax1.imshow(cam_img)
#ax1.imshow(skimage.transform.resize(cam_img, img.shape[1:]), alpha=0.5, cmap='jet')
#ax1.axis('off')
#ax1.set_aspect(1)
#ax2.bar(index, ps[idx], width=0.8, align='center', color='darkred')
#ax2.bar(gt_idx, ps[idx][gt_idx], width=0.8, align='center', color='green')
#ax2.set_aspect(5)


#for out in outputs:
#    print(out.shape)
