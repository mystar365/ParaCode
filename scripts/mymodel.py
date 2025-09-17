# Standard library
import os
import random
import time
import warnings
from collections import namedtuple
from typing import Type, Union, List, Optional
import math
import pdb

# Third-party libraries
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.nn import (
    Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid,
    Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d,
    Sequential, Module, Parameter, Conv1d, MaxPool1d, AdaptiveAvgPool1d
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error
)

# Other utilities
from tqdm import tqdm
import torch.cuda

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mymodel: important function and class


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):  
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_diff = torch.abs(input - target)
        loss = torch.where(abs_diff < self.delta,
                          0.5 * (abs_diff ** 2),
                          self.delta * abs_diff - 0.5 * (self.delta ** 2))
        return torch.sum(loss) 


class WHuberLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(WHuberLoss, self).__init__()
        self.delta = delta
    def forward(self, input, target, weight):
        
        weights = weight 
        abs_diff = torch.abs(input - target)  
        
        loss = torch.where(
            abs_diff < self.delta,
            0.5 * weights * (abs_diff ** 2), 
            weights * self.delta * abs_diff - 0.5 * (self.delta ** 2) 
        )
        
        return torch.sum(loss)  
         
class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool1d(1)
        self.fc1 = Conv1d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.fc1_new = Conv1d(
            1, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = PReLU()
        self.fc2 = Conv1d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)  
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool1d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv1d(in_channel, depth, 1, stride, bias=False), BatchNorm1d(depth))
        self.res_layer = Sequential(
            BatchNorm1d(in_channel),
            Conv1d(in_channel, depth, 3, 1, 1, bias=False), PReLU(depth),
            Conv1d(depth, depth, 3, stride, 1, bias=False), BatchNorm1d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool1d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv1d(in_channel, depth, 1, stride, bias=False),
                BatchNorm1d(depth))
        self.res_layer = Sequential(
            BatchNorm1d(in_channel),
            Conv1d(in_channel, depth, 3, 1, 1, bias=False),
            PReLU(depth),
            Conv1d(depth, depth, 3, stride, 1, bias=False),
            BatchNorm1d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 66:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=4),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=4),
            get_block(in_channel=256, depth=512, num_units=4)
        ]
    elif num_layers ==34 :
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 98:
        blocks = [get_block(in_channel=64, depth=64, num_units=6),
                  get_block(in_channel=64, depth=128, num_units=6),
                  get_block(in_channel=128, depth=256, num_units=6),
                  get_block(in_channel=256, depth=512, num_units=6)
                  ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=1),
            get_block(in_channel=64, depth=128, num_units=1),
            get_block(in_channel=128, depth=256, num_units=1),
            get_block(in_channel=256, depth=512, num_units=1)
        ] 
    return blocks


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, num_class=1, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [18,34, 66, 98, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv1d(1, 64, 3, 1, 1, bias=False),
                                      BatchNorm1d(64),
                                      PReLU(64))

        self.gru_layer = nn.GRU(245,245,batch_first= True,bidirectional=False)
        
        self.output_layer = Sequential(
        nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=1, bias=False),
        BatchNorm1d(1),  # (batch_size, 1, length)
        PReLU(1),
        Dropout(drop_ratio),
        Flatten(),  # (batch_size, 1 * length=245)
    
        Linear(245, 120),  
        PReLU(),          
        Dropout(drop_ratio), 
    
        Linear(120, num_class))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    def forward(self, x):
        x = x.view(x.size(0),1,3909)
        x = self.input_layer(x) 
        x = self.body(x)
        Z, h_final = self.gru_layer(x)  
        x = h_final.transpose(0,1) 
        x = self.output_layer(x)  
        return x
        
if __name__ == "__main__":
    print('model-definition')
    