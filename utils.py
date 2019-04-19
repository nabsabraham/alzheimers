#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:30:02 2019

@author: nabila
"""
import torch
import numpy as np 
import matplotlib.pyplot as plt 

def view_classify(img, ps, gt):
    ''' Function for viewing an image and it's predicted classes.
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
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
    
    
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
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