import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from config import *

class ResNet18(nn.Module):
    def __init__(self, out_features, pretrained=True, freeze_pretrained=True):
        super().__init__()

        self.name = 'ResNet18'

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None   # pre-trained weights
        self.resnet = models.resnet18(weights=weights)
        if freeze_pretrained:   # freeze the model's weights
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        self.fcs = nn.Sequential(   # to get final output logit
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fcs(x)

        return x
    
class MobileNet_V3_Small(nn.Module):
    def __init__(self, out_features, pretrained=True, freeze_pretrained=True):
        super().__init__()

        self.name = 'MobileNet_V3_Small'

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None   # pre-trained weights
        self.resnet = models.mobilenet_v3_small(weights=weights)
        if freeze_pretrained:   # freeze the model's weights
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.fcs = nn.Sequential(   # to get final output logit
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fcs(x)

        return x