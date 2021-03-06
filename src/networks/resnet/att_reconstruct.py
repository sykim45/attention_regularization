import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

from networks.utils import conv3x3, BasicBlock, Bottleneck
from networks.subnet.dropout import ReconstructDropout

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]


class ResNet_att_reconstruct(nn.Module):
    def __init__(self, depth, num_classes, dropout_rate, is_imagenet):
        super(ResNet_att_reconstruct, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.drop_rate = dropout_rate
        self.is_imagenet = is_imagenet

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.dropout = ReconstructDropout(self.linear.weight, self.linear.bias, self.drop_rate, device=torch.device('cuda'))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, rand_idx=None):
        if rand_idx is not None:
            x_f = x[rand_idx,:,:,:]
            out_f = F.relu(self.bn1(self.conv1(x_f)))
            out_f = self.layer1(out_f)
            out_f = self.layer2(out_f)
            out_f= self.layer3(out_f)
            out_f = self.layer4(out_f)
            gap_f = F.avg_pool2d(out_f, 8) if self.is_imagenet else F.avg_pool2d(out_f, 4)
            gap_f = gap_f.view(gap_f.size(0), -1)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            gap = F.avg_pool2d(out, 8) if self.is_imagenet else F.avg_pool2d(out, 4)
            gap = gap.view(gap.size(0), -1)
            out = self.dropout(gap, gap_f, self.linear(gap), self.linear(gap_f))
            #out = self.linear(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            gap = F.avg_pool2d(out, 8) if self.is_imagenet else F.avg_pool2d(out, 4)
            gap = gap.view(gap.size(0), -1)
            out = self.linear(gap)
        return out
