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
train_loader, val_loader = get_dataloaders(data_path, val_split=0.2, batch_size=16, shuffle=True)

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
                
model = newVGG(n_classes=3)
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
        
#        soft_preds, preds = torch.max(log_ps, 1)
#        acc = torch.sum(torch.abs((labels-preds)**1))
#        train_acc.append(acc.item())
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
            
#            soft_preds, preds = torch.max(log_ps, 1)
#            acc = torch.sum(torch.abs((labels-preds)**1))
#            val_acc.append(acc.item())                
            val_loss.append(loss.item())
                
    print('[%d]/[%d] Train Loss:%.4f\t Train Acc:%.4f\t Val Loss:%.4f\t Val Acc: %.4f'
          % (epoch+1, epochs, 
             np.mean(train_loss),  
             np.mean(train_acc),  
             np.mean(val_loss),  
             np.mean(val_acc)))

    early_stopping(np.average(val_loss), model)
    
    if early_stopping.early_stop:
        print("Early stopping at epoch: ", epoch)
        break

outputs=[]
def hook(module, input, output):    
    outputs.append(output)
       
model.features[14].register_forward_hook(hook)

images, labels = next(iter(val_loader))
images = images.to(device)
prediction = model(images).cpu().detach()
_, preds = torch.max(prediction,1)
#pred_probabilities = F.softmax(prediction).data.squeeze()
#topk(prediction,1)

fc_weights = model._modules.get('classifier').parameters()
for i in range(3): weights = next(iter(fc_weights))
class_idx = topk(prediction,1)[1]

overlay = utils.getCAM(outputs[-3], weights, class_idx.int() )
feature_conv = outputs[0][1,:256,:,:].cpu().detach()
weight_fc = weights.cpu().detach()
class_idx = class_idx[1]

nc, h, w = feature_conv.shape
cam = torch.matmul(weight_fc[class_idx],feature_conv.reshape((nc, h*w)))
cam = cam.reshape(h, w)
cam = cam.numpy()
cam = cam - np.min(cam)
cam_img = cam / np.max(cam)

plt.imshow(images[1].cpu().detach().permute(1,2,0).numpy())
plt.imshow(skimage.transform.resize(cam_img, (150,150)), alpha=0.5, cmap='jet')


idx = 4
classes = ['AD', 'MCI', 'Normal']
img = images[idx].unsqueeze(0)

ps = torch.exp(model(img))
gt = classes[int(labels[idx])]
utils.view_classify(img, ps, gt)

#ps = ps.cpu().detach().numpy().squeeze()
#img = img.cpu().detach().squeeze(0)
#
#fig, (ax1, ax2) = plt.subplots(figsize=(5,5), ncols=2)
#ax1.imshow(img.permute(1,2,0).cpu().detach().numpy().squeeze())
#ax1.set_title('True label =' + gt)
#ax1.axis('off')
#ax2.barh(np.arange(3), ps)
#ax2.set_aspect(0.1)
#ax2.set_yticks(np.arange(3))
#ax2.set_yticklabels(['AD',
#                     'MCI',
#                     'Normal'], size='medium');
#ax2.set_title('Class Probability')
#ax2.set_xlim(0, 1.1)

for out in outputs:
    print(out.shape)
