import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################
# >>> Convolution Utils
######################################################
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class CAMBlockDropout(nn.Module):
    def __init__(self, planes, num_classes, drop_rate, is_imagenet):
        super(CAMBlockDropout, self).__init__()
        self.is_imagenet = is_imagenet
        self.p = drop_rate
        self.linear = nn.Linear(planes, num_classes)
        self.weight = self.linear.weight

    def _discriminative_idx(self, outputs):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        #k = round(self.weight.size(0) * self.p)
        #topk_idx = torch.topk(cam_weight, k)[1]
        #return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx
        return idx[:, 0]

    def forward(self, input):
        gap = F.avg_pool2d(input, 8) if self.is_imagenet else F.avg_pool2d(input, 4)
        # print('gap:{}'.format(gap.size()))
        gap = gap.view(gap.size(0), -1)
        cam_idx = self._get_cam_weight(self.linear(gap))
        input[cam_idx] = F.dropout(input[cam_idx], self.p)
        return input


######################################################
# >>> ResNet blocks
######################################################
# >>> basic
class BasicBlockDropout(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, drop_rate, stride=1):
        super(BasicBlockDropout, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(planes)
        self.dropout = nn.Dropout(p = drop_rate)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


# >>> bottleneck
class BottleneckDropout(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu3 = nn.ReLU(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.dropout = BlockDropout(0.3, planes, 200)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)

        return out


class BlockDropout(nn.Module):
    def __init__(self, drop_rate, plane, num_classes):
        super(BlockDropout, self).__init__()
        self.p = drop_rate
        self.lin = nn.Linear(plane*plane, num_classes)
        self.weight = self.lin.weight
        self.device = torch.device('cuda')

    def forward(self, x):
        out = F.avg_pool2d(x, 16)
        out = out.view(out.size(0), -1)
        idx, topk_idx = self.discriminative_idx_(out, self.lin(out))
        #print(x.size())
        mask = torch.empty(x.size()).fill_(1).to(self.device)
        mask[idx, topk_idx] = 0
        return x * mask

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def discriminative_idx_(self, feature, outputs):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        #cam_diff = cam_weight[:,1:] - cam_weight[:,:-1]
        k = round(self.weight.size(0) * self.p)
        topk_idx = torch.topk(cam_weight, k)[1]
        return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx


######################################################
# >>> Wide-Resnet blocks
######################################################
class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, conv_bias=False):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=conv_bias)
        self.relu1 = nn.ReLU(in_planes)
        self.dropout_rate = dropout_rate
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.relu2 = nn.ReLU(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=conv_bias),
            )

    def forward(self, x):
        #out = F.dropout(self.conv1(self.relu1(self.bn1(x))), p=self.dropout_rate, training=self.training)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = F.dropout(out, self.dropout_rate, training=self.trainig)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += self.shortcut(x)

        return out
######################################################
