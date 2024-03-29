#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:30:02 2019

@author: nabila
"""
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision.utils as vutils 

def multi_accuracy(preds, labels):
    labels = labels.float()
    ps = torch.exp(preds)
    _, top_ps = torch.topk(ps,1)
    top_ps = top_ps.reshape(labels.shape).float()
    acc = torch.mean(torch.eq(top_ps, labels).float()) 
    return acc
        
def binary_accuracy(preds, labels):
    labels = labels.float()
    p = (preds >= 0.5)
    p = p.float()
    acc = torch.mean(torch.eq(p,labels).float())
    return acc

# udacity
def view_classify(img, ps, gt):
    ''' Function for viewing an image and it's predicted classes as a bar graph
    '''
    ps = ps.cpu().detach().numpy().squeeze()
    img = img.cpu().detach().squeeze(0)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(5,5), ncols=2)
    ax1.imshow(img.permute(1,2,0).cpu().detach().numpy().squeeze())
    ax1.set_title('True label =' + gt)
    
    ax1.axis('off')
    ax2.barh(np.arange(3), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(3))
    ax2.set_yticklabels(['AD',
                         'MCI',
                         'Normal'], size='medium');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

#http://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html
#adapted version
def get_cam(features_np, weights_np, top_idx):
    nc, h, w = features_np.shape
    F = features_np.reshape((nc, h*w))
    W = weights_np[top_idx]
    
    cam = np.dot(W,F)
    cam = np.reshape(cam, (h,w))
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img
    
def plot_hist(epochs, train, val, data_name):
    plt.figure(dpi=150)
    e = np.arange(0, epochs+1,1)
    plt.plot(e, train, label='Train'+ " " + str(data_name))
    plt.plot(e, val, label='Val'+ " " + str(data_name))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(str(data_name) + " Values")
    plt.grid()
    plt.title(str(data_name) + " Results")

def plot_grid(torch_preds, name):
    torch_preds = torch_preds.cpu().detach()
    grid_img = vutils.make_grid(torch_preds, nrow=4)
    plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(str(name))
    
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased %.4f --> %.4f).  Saving model ...' % (self.val_loss_min, val_loss))
        name = 'checkpoint.pt'
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss

# https://github.com/pytorch/examples/blob/537f6971872b839b36983ff40dafe688276fe6c3/imagenet/main.py#L340
# https://github.com/pytorch/examples/blob/537f6971872b839b36983ff40dafe688276fe6c3/imagenet/main.py#L234

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')