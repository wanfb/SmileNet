'''
    implement SmileNet
    @author: Fangbin Wan
    @date: 2019.1.30
'''
from __future__ import absolute_import
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from torch.autograd import Variable

class Smile_Net(nn.Module):
    def __init__(self, num_classes=0):
        super(Smile_Net, self).__init__()
        self.num_classes = num_classes
        self.Block1=torch.nn.Sequential()
        self.Block1.add_module("conv1",torch.nn.Conv2d(1, 32, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1)))
        self.Block1.add_module("drop1",torch.nn.Dropout(p=0.5, inplace=False))
        self.Block1.add_module("relu1",torch.nn.ReLU(inplace=True))
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.Block2=torch.nn.Sequential()
        self.Block2.add_module("conv2",torch.nn.Conv2d(32, 32, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1)))
        self.Block2.add_module("drop2",torch.nn.Dropout(p=0.25, inplace=False))
        self.Block2.add_module("relu2",torch.nn.ReLU())
        
        self.pool2 = torch.nn.AvgPool2d(kernel_size = (2, 2), stride=(2, 2))

        self.Block3=torch.nn.Sequential()
        self.Block3.add_module("conv3",torch.nn.Conv2d(32, 32, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1)))
        self.Block3.add_module("drop3",torch.nn.Dropout(p=0.3, inplace=False))
        self.Block3.add_module("relu3",torch.nn.ReLU())
        
        self.pool3 = torch.nn.AvgPool2d(kernel_size = (2, 2), stride=(2, 2))

        self.fc= torch.nn.Linear(8192, self.num_classes)

        
    def forward(self, x):
        x = self.Block1(x)
        x = self.pool1(x)
        x = self.Block2(x)
        x = self.pool2(x)
        x = self.Block3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.001)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                init.constant(m.bias, 0)

def SmileNet(**kwargs):
    model = Smile_Net(**kwargs)
    return model



