#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:43:36 2019

This script gets the datasets with labels using ImageFolder
Running main will show a demo of how to use the get_dataloaders function

@author: nabila
"""
import torch
import torchvision 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

random_seed = 42
     
    
def get_dataloaders(data_path, val_split, batch_size, shuffle=True):
    t = transforms.Compose([transforms.Resize((150,150)), 
                            transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(root=data_path,transform=t)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
      
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
        
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader


if __name__ == '__main__':
    
    data_path = "/home/nabila/Desktop/datasets/ADNI/Processed_32_per_subject"
    val_split = 0.2
    shuffle = True
    batch_size = 32
    classes = ['AD','MC','Normal']
    train_loader, val_loader = get_dataloaders(data_path, val_split=0.2, batch_size=32,shuffle=True)
    images, labels = next(iter(val_loader))

    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(classes[labels[idx]])

