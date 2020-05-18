import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import sys
import numpy as np

from networks.utils import conv3x3, wide_basic
from networks.subnet.dropout import AttentionDropout

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, conv_bias, dropout_rate, num_classes, is_imagenet=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.is_imagenet = is_imagenet

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0],stride=1,bias=conv_bias)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, conv_bias=conv_bias)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, conv_bias=conv_bias)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, conv_bias=conv_bias)
        self.bn1 = nn.BatchNorm2d(nStages[3]) # 20200114 changed here
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, conv_bias):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, conv_bias))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 16) if self.is_imagenet else F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
