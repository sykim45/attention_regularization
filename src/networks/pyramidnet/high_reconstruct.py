import torch
import torch.nn as nn
import math
from networks.utils import conv3x3, PyramidBasicBlock, PyramidBottleneck
from networks.subnet.dropout import DetectionMix

class PyramidNet_high_reconstruct(nn.Module):

    def __init__(self, depth, alpha, num_classes, drop_rate, is_imagenet=False, bottleneck=False):
        super(PyramidNet_high_reconstruct, self).__init__()
        self.is_imagenet = is_imagenet
        self.drop_rate = drop_rate
        if self.is_imagenet is False:
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = PyramidBottleneck
            else:
                n = int((depth - 2) / 6)
                block = PyramidBasicBlock

            self.addrate = alpha / (3 * n * 1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.linear = nn.Linear(self.final_featuremap_dim, num_classes)
            self.dropout = DetectionMix(self.linear.weight, self.linear.bias, self.drop_rate,
                                        device=torch.device('cuda'))

        elif self.is_imagenet:
            blocks = {18: PyramidBasicBlock, 34: PyramidBasicBlock, 50: PyramidBottleneck, 101: PyramidBottleneck, 152: PyramidBottleneck, 200: PyramidBottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                      200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = PyramidBottleneck
                    temp_cfg = int((depth - 2) / 12)
                else:
                    blocks[depth] = PyramidBasicBlock
                    temp_cfg = int((depth - 2) / 8)

                layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.inplanes = 64
            self.addrate = alpha / (sum(layers[depth]) * 1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7)
            self.linear = nn.Linear(self.final_featuremap_dim, num_classes)
            self.dropout = DetectionMix(self.linear.weight, self.linear.bias, self.drop_rate,
                                        device=torch.device('cuda'))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x, rand_idx=None):
        if self.is_imagenet is False:
            if rand_idx is not None:
                x_f = x[rand_idx, :, :, :]
                x_f = self.conv1(x_f)
                x_f = self.bn1(x_f)
                x_f = self.layer1(x_f)
                x_f = self.layer2(x_f)
                x_f = self.layer3(x_f)
                x_f = self.bn_final(x_f)
                x_f = self.relu_final(x_f)
                gap_f = self.avgpool(x_f)
                gap_f = gap_f.view(gap_f.size(0), -1)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.bn_final(x)
                x = self.relu_final(x)
                gap = self.avgpool(x)
                gap = gap.view(gap.size(0), -1)
                x, ratio = self.dropout(gap, gap_f, self.linear(gap), self.linear(gap_f))
                return x, ratio
            else:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.bn_final(x)
                x = self.relu_final(x)
                gap = self.avgpool(x)
                gap = gap.view(gap.size(0), -1)
                return self.linear(gap)

        elif self.is_imagenet:
            if rand_idx is not None:
                x_f = x[rand_idx, :, :, :]
                x_f = self.conv1(x_f)
                x_f = self.bn1(x_f)
                x_f = self.relu(x_f)
                x_f = self.maxpool(x_f)
                x_f = self.layer1(x_f)
                x_f = self.layer2(x_f)
                x_f = self.layer3(x_f)
                x_f = self.layer4(x_f)
                x_f = self.bn_final(x_f)
                x_f = self.relu_final(x_f)
                gap_f = self.avgpool(x_f)
                gap_f = gap_f.view(gap_f.size(0), -1)

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.bn_final(x)
                x = self.relu_final(x)
                gap = self.avgpool(x)
                gap = gap.view(gap.size(0), -1)
                x, ratio = self.dropout(gap, gap_f, self.linear(gap), self.linear(gap_f))
                return x, ratio
            else:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.bn_final(x)
                x = self.relu_final(x)
                gap = self.avgpool(x)
                gap = gap.view(gap.size(0), -1)
                return self.linear(gap)