#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:47:11 2019

Binary classification script - using CrossEntropy 
@author: nabila
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:43:36 2019

@author: nabila
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import skimage

from data import get_dataloaders
import utils
import models
import os 

data_path = "/home/nabila/Desktop/datasets/ADNI/MCIvNormal"
val_split = 0.2
batch_size = 32
epochs = 350
lr = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ['AD', 'Normal']

train_loader, val_loader = get_dataloaders(data_path, 
                                           val_split=val_split, 
                                           batch_size=batch_size, 
                                           shuffle=True)
                
model = models.VGG512(n_classes=2)

model = model.to(device)
criterion = nn.NLLLoss()
opt = optim.Adam(model.parameters(), lr=lr)
early_stopping = utils.EarlyStopping(patience=8, verbose=True)

epoch_train_loss = []
epoch_val_loss = []
epoch_train_acc = []
epoch_val_acc = []

print('Training...')
for epoch in range(epochs):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for images, labels in train_loader:
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        log_ps = model(images)      
        loss = criterion(log_ps, labels)
        
        acc = utils.multi_accuracy(log_ps, labels)
        train_acc.append(acc.item())
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for images, labels in val_loader:
            model.eval()
            images = images.to(device)
            labels = labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            
            acc = utils.multi_accuracy(log_ps, labels)
            val_acc.append(acc.item())                
            val_loss.append(loss.item())
            
    print('')                
    print('[%d]/[%d] Train Loss:%.4f\t Train Acc:%.4f\t Val Loss:%.4f\t Val Acc: %.4f'
          % (epoch+1, epochs, 
             np.mean(train_loss),  
             np.mean(train_acc),  
             np.mean(val_loss),  
             np.mean(val_acc)))

    epoch_train_loss.append(np.mean(train_loss))
    epoch_val_loss.append(np.mean(val_loss))
    epoch_train_acc.append(np.mean(train_acc))
    epoch_val_acc.append(np.mean(val_acc))

    early_stopping(np.average(val_loss), model)
    
    if early_stopping.early_stop:
        print("Early stopping at epoch: ", epoch)
        break

utils.plot_hist(epoch, np.array(epoch_train_loss), np.array(epoch_val_loss), "Loss")
plt.title('Lowest loss = %.4f' % epoch_val_loss[-1])

utils.plot_hist(epoch, np.array(epoch_train_acc), np.array(epoch_val_acc), "Acc")
plt.title('Best accuracy = %.4f' % epoch_val_acc[-1])

# Load up the best model and view preds
model.load_state_dict(torch.load('checkpoint.pt'))

outputs=[]
def hook(module, input, output):    
    outputs.append(output)

model.features.register_forward_hook(hook)

X,Y = next(iter(val_loader))
X = X.to(device)
Y = Y.to(device)
ps = torch.exp(model(X)).detach().cpu()
soft_ps, top_classes = torch.max(ps,1)
   
fc_weights = model._modules.get('classifier').parameters()
for i in range(3): weights = next(iter(fc_weights))

# For each random idx in the validation set, plot the CAM
idx = np.random.randint(batch_size)
img = X[idx] 
gt_idx = int(Y[idx])
gt_class = classes[gt_idx]
pred_class = classes[top_classes[idx]]
pred_ps = soft_ps[idx]

features_np = outputs[-1].cpu().detach().numpy()[idx]
weights_np = weights.cpu().detach().numpy()

cam_img = utils.get_cam(features_np, weights_np, top_classes[idx])
plt.figure()
plt.imshow(images[idx].cpu().detach().permute(1,2,0).numpy())
plt.imshow(skimage.transform.resize(cam_img, img.shape[1:]), alpha=0.5, cmap='jet')
plt.title('Pred = %s (%.4f) | GT = %s' % (pred_class, pred_ps, gt_class))
plt.axis('off')

utils.view_classify(img, ps[idx], gt_class)

# ==============================================
# Uncomment below to write the results to a dir: 
# ==============================================
'''
write_to = "/home/nabila/Desktop/github/alzheimers/Training Results/MCIvNormal"

for idx, img in enumerate(zip(X,Y)):
    img = X[idx].cpu().detach().permute(1,2,0).numpy()
    gt_idx = int(Y[idx])
    gt_class = classes[gt_idx]
    pred_class = classes[top_classes[idx]]
    pred_ps = soft_ps[idx]
    
    features_np = outputs[-1].cpu().detach().numpy()[idx]
    weights_np = weights.cpu().detach().numpy()
    
    cam_img = utils.get_cam(features_np, weights_np, top_classes[idx])
    plt.figure()
    plt.imshow(X[idx].cpu().detach().permute(1,2,0).numpy())
    plt.imshow(skimage.transform.resize(cam_img, img.shape[0:2]), alpha=0.5, cmap='jet')
    plt.title('Pred = %s (%.4f) | GT = %s' % (pred_class, pred_ps, gt_class))
    plt.axis('off')
    fname = str(idx)
    plt.savefig(os.path.join(write_to, fname))
'''
