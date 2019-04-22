#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:08:05 2019

@author: nabila
"""
import torch.nn as nn
import torchvision.models as models 

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
 
class binaryVGG(nn.Module):
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
                                   nn.Sigmoid())
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
class VGG512(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).cuda()
        self.features = self.vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(nn.Linear(8192,512), 
                                   nn.ReLU(),
                                   nn.Dropout(0.5), 
                                   nn.Linear(512,n_classes), 
                                   nn.LogSoftmax(dim=1))
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x