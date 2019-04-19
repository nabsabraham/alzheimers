#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:51:48 2019

@author: nabila
"""

from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt 


image = Image.open("cat.jpg")
imshow(image)

# Imagenet mean/std

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor, 
# and normalize to -1..1 with mean/std for ImageNet

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([transforms.Resize((224,224))])

tensor = preprocess(image)
prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

resnet = models.resnet18(pretrained=True)
resnet.cuda()
resnet.eval()

class SaveFeatures():
    features=None
    
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output): 
        self.features = ((output.cpu()).data).numpy()
        
    def remove(self): 
        self.hook.remove()

final_layer = resnet._modules.get('layer4')
activated_features = SaveFeatures(final_layer)
prediction = resnet(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()
topk(pred_probabilities,1)

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


weight_softmax_params = list(resnet._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

weight_softmax.shape
class_idx = topk(pred_probabilities,1)[1].int()
overlay = getCAM(activated_features.features, weight_softmax, class_idx )
imshow(overlay[0], alpha=0.5, cmap='jet')

plt.figure()
imshow(display_transform(image))
imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')




