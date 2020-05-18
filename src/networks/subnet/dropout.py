import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import sys
import numpy as np

from networks.utils import conv3x3

class AttentionDropout(nn.Module):
    def __init__(self, planes, conv_bias, is_imagenet=False):
        super(AttentionDropout, self).__init__()
        self.key_conv = nn.Conv2d(planes, planes, kernel_size=1, padding=0, stride=1, bias=conv_bias)
        self.query_conv = nn.Conv2d(planes, planes, kernel_size=1, padding=0, stride=1, bias=conv_bias)
        self.softmax = nn.Softmax(dim=1)
        self.is_imagenet=is_imagenet

    def forward(self, x, dropout_ratio):
        key = self.key_conv(x)
        query = self.query_conv(x)
        pool_factor = 16 if self.is_imagenet else 8
        value = F.avg_pool2d(x, pool_factor)
        value = value.view(value.size(0), -1)

        attention = key * query
        attention = F.avg_pool2d(attention, pool_factor)
        attention = attention.view(attention.size(0), -1)
        attention = self.softmax(attention)

        TopK = int((1-dropout_ratio/2) * attention.shape[1])
        BottomK = int((dropout_ratio/2) * attention.shape[1])
        # sort attention, and drop (num_nodes * dropout_ratio) nodes
        argsort_idx = torch.argsort(attention, dim=1, descending=True)
        value[argsort_idx > TopK] = 0.
        value[argsort_idx < BottomK] = 0.

        return value


class CAMDropout(nn.Module):
    __constants__ = ['p']

    def __init__(self, weight_matrix, bias, drop_rate, device):
        super(CAMDropout, self).__init__()
        self.weight = weight_matrix #np.squeeze(weight_matrix.cpu().data.numpy())
        self.device = device
        self.p = drop_rate
        self.bias = bias

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def _discriminative_idx(self, feature, outputs):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        #cam_diff = cam_weight[:,1:] - cam_weight[:,:-1]
        k = round(self.weight.size(1) * 0.5)
        topk_idx = torch.topk(cam_weight, k)[1]
        #perm = torch.randperm(topk_idx.size(1)).cuda()
        #k_idx = int(self.weight.size(1)*self.p)
        return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def forward(self, features, output):
        if self.training:
            idx, topk_idx = self._discriminative_idx(features, output)
            mask = torch.empty(self.weight.size()).fill_(1).to(self.device)
            mask_b = torch.empty(self.bias.size()).fill_(1).to(self.device)
            #mask[idx, topk_idx] = 0
            #mask[idx, topk_idx] = torch.bernoulli(torch.empty(mask[idx,topk_idx].size()).fill_(1-self.p)).to(self.device)
            print(torch.mean(self.weight[idx,topk_idx], dim=1).size())
            mask[idx, topk_idx] = F.dropout(mask[idx, topk_idx], self.p)
            mask_b[idx] = 0
            res = F.linear(features, self.weight * mask, self.bias * mask_b)
            return res
        else:
            return output


class ReconstructDropout(nn.Module):
    """
    Cutmix on latent features
    """
    def __init__(self, weight_matrix, bias, drop_rate, device):
        super(ReconstructDropout, self).__init__()
        self.p = drop_rate
        self.weight = weight_matrix
        self.device = device
        self.bias = bias
        #self.rand_idx = torch.randperm(features.size()[0]).cuda() if np.random.binomial(1, self.p)
        #self.bias = bias

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def discriminative_idx_(self, feature, outputs):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        k = round(self.weight.size(0) * self.p)
        topk_idx = torch.topk(cam_weight, k)[1]
        return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def forward(self, features, features_f, output, output_f):
        idx, topk_idx = self.discriminative_idx_(features, output)
        idx_f, topk_idx_f = self.discriminative_idx_(features_f, output_f)
        mask = self.weight.clone().detach() #torch.empty(self.weight.size()).fill_(1).to(self.device)
        mask_b = self.bias.clone().detach() #torch.empty(self.bias.size()).fill_(1).to(self.device)
        #features[:, topk_idx] = features_f[:, topk_idx_f]
        mask[idx, topk_idx] = mask[idx_f, topk_idx_f]
        mask_b[idx] = mask_b[idx_f]
        return F.linear(features, mask, mask_b)
        #else:
         #   return features


class StochasticReconstruct(nn.Module):
    # Detect sweet spot by normalizing CAM weight by Gaussian dist
    def __init__(self, weight_matrix, bias, drop_rate, device):
        super(StochasticReconstruct, self).__init__()
        self.weight = weight_matrix
        self.device = device
        self.p = drop_rate
        self.bias = bias

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def _discriminative_idx(self, feature, outputs):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        #cam_diff = cam_weight[:,1:] - cam_weight[:,:-1]
        k = round(self.weight.size(1) * self.p)
        topk_idx = torch.topk(cam_weight, k)[1]
        #perm = torch.randperm(topk_idx.size(1)).cuda()
        #k_idx = int(self.weight.size(1)*self.p)
        return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def forward(self, features, features_f, output, output_f):
        #features_f = features[rand_idx, :]
        #output_f = output[rand_idx, :]
        idx, topk_idx = self._discriminative_idx(features, output)
        idx_f, topk_idx_f = self._discriminative_idx(features_f, output_f)
        mask = self.weight.clone().detach()
        mask_b = self.bias.clone().detach()

        mask[idx, topk_idx] = mask[idx_f, topk_idx_f]
        mask_b[idx] = mask_b[idx_f]
        mask = F.dropout(mask, self.p)
        #print(mask[idx,topk_idx].size(), mask_f[idx_f,topk_idx_f].size())
        #features[idx, topk_idx] = features_f[idx_f, topk_idx_f]
        return F.linear(features, mask, mask_b)



class DetectionMix(nn.Module):
    def __init__(self, weight_matrix, bias, drop_rate, device, mode='max'):
        super(DetectionMix, self).__init__()
        self.weight = weight_matrix
        self.device = device
        self.p = drop_rate
        self.bias = bias
        self.mode = mode

    def extra_repr(self):
        return 'p={}, mode={}'.format(self.p, self.mode)

    def _discriminative_idx(self, features, outputs, rate=None):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        ratio = self._get_ratio(cam_weight) if rate is None else rate
        high_k = int(self.weight.size(1) * ratio)
        #low_k = self.weight.size(1) - high_k
        #topk_idx = torch.topk(cam_weight, low_k, largest=False)[1] if reverse else torch.topk(cam_weight, high_k)[1]
        topk_idx = torch.topk(cam_weight, high_k)[1]
        if rate is None:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx, ratio
        else:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def _get_ratio(self, cam_weight):
        cam_mean = torch.mean(cam_weight, dim=1)
        MADv = torch.sum((cam_weight - cam_mean.reshape((cam_weight.size(0), 1))).abs(), dim=1) / cam_weight.size(1)
        RMSD = torch.sqrt(torch.sum((cam_weight - cam_mean.reshape((cam_weight.size(0), 1))) ** 2, dim=1) / cam_weight.size(1))
        cam_thrs = cam_mean + MADv #RMSD
        ix = torch.where(cam_weight > cam_thrs.reshape(cam_thrs.size(0), 1))
        cnt = torch.from_numpy(np.unique(ix[0].cpu(), return_counts=True)[1]).float()
        max = torch.max(cnt)
        mean = torch.mean(cnt)
        mode = torch.mode(cnt)[0].data
        return max / cam_weight.size(1)

    def forward(self, features, features_f, output, output_f):
        idx, topk_idx, mix_ratio = self._discriminative_idx(features, output)
        idx_f, topk_idx_f = self._discriminative_idx(features_f, output_f, mix_ratio)

        mask = self.weight.clone().detach()
        mask_b = self.bias.clone().detach()
        mask[idx, topk_idx] = mask[idx_f, topk_idx_f]
        mask_b[idx] = mask_b[idx_f]
        #print(mask[idx,topk_idx].size(), mask_f[idx_f,topk_idx_f].size())
        #features[idx, topk_idx] = features_f[idx_f, topk_idx_f]
        return F.linear(features, mask, mask_b), mix_ratio



class RmNoiseMix(nn.Module):
    def __init__(self, weight_matrix, bias, drop_rate, device, mode='mean'):
        super(RmNoiseMix, self).__init__()
        self.weight = weight_matrix
        self.device = device
        self.p = drop_rate
        self.bias = bias
        self.mode = mode

    def extra_repr(self):
        return 'p={}, mode={}'.format(self.p, self.mode)

    def _discriminative_idx(self, features, outputs, rate=None):
        h_x = F.softmax(outputs, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        # idx[:,0].size())
        cam_weight = self.weight[idx[:, 0], ]
        ratio = self._get_ratio(cam_weight) if rate is None else rate
        low_k = int(self.weight.size(1) * ratio)
        #low_k = self.weight.size(1) - high_k
        #topk_idx = torch.topk(cam_weight, low_k, largest=False)[1] if reverse else torch.topk(cam_weight, high_k)[1]
        topk_idx = torch.topk(cam_weight, low_k, largest=False)[1]
        if rate is None:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx, ratio
        else:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def _get_ratio(self, cam_weight):
        cam_mean = torch.mean(cam_weight, dim=1)
        MADv = torch.sum((cam_weight - cam_mean.reshape((cam_weight.size(0), 1))).abs(), dim=1) / cam_weight.size(1)
        cam_thrs = cam_mean - MADv
        ix = torch.where(cam_weight < cam_thrs.reshape(cam_thrs.size(0), 1))
        cnt = torch.from_numpy(np.unique(ix[0].cpu(), return_counts=True)[1]).float()
        max = cnt.max()
        mean = torch.mean(cnt)
        mode = torch.mode(cnt)
        return max / cam_weight.size(1)

    def forward(self, features, features_f, output, output_f):
        idx, topk_idx, mix_ratio = self._discriminative_idx(features, output)
        idx_f, topk_idx_f = self._discriminative_idx(features_f, output_f, mix_ratio)

        mask = self.weight.clone().detach()
        mask_b = self.bias.clone().detach()
        mask[idx, topk_idx] = mask[idx_f, topk_idx_f]
        mask_b[idx] = mask_b[idx_f]
        #print(mask[idx,topk_idx].size(), mask_f[idx_f,topk_idx_f].size())
        #features[idx, topk_idx] = features_f[idx_f, topk_idx_f]
        return F.linear(features, mask, mask_b), mix_ratio





"""
class FeatureMapMix(nn.Module):
    def __init__(self, weight_matrix, bias, drop_rate, device):
        super(FeatureMapMix, self).__init__()
        self.weight = weight_matrix
        self.device = device
        self.p = drop_rate
        self.bias = bias

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def _discriminative_idx(self, features, outputs, rate=None):
       # h_x = F.softmax(outputs, dim=1).data.squeeze()
        #_, idx = h_x.sort(0, True)
        # idx[:,0].size())
        #cam_weight = self.weight #[idx[:, 0], ]
        ratio = self._get_ratio(self.weight) if rate is None else rate
        high_k = int(self.weight.size(1) * ratio)
        #low_k = self.weight.size(1) - high_k
        #topk_idx = torch.topk(cam_weight, low_k, largest=False)[1] if reverse else torch.topk(cam_weight, high_k)[1]
        topk_idx = torch.topk(self.weight, high_k)[1]
        if rate is None:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx, ratio
        else:
            return idx[:, 0].reshape((topk_idx.size(0),1)), topk_idx

    def _get_ratio(self, weight):
        mean = torch.mean(weight, dim=0)
        MADv = torch.sum((weight - mean.reshape((1, weight.size(1)))).abs(), dim=1) / weight.size(0)
        thrs = mean + MADv
        ix = torch.where(weight > thrs.reshape(1, thrs.size(1)))
        cnt = torch.from_numpy(np.unique(ix[0].cpu(), return_counts=True)[1])
        max = cnt.max()
        return max / weight.size(0)

    def forward(self, features, features_f, output, output_f):
        idx, topk_idx, mix_ratio = self._discriminative_idx(features, output)
        idx_f, topk_idx_f = self._discriminative_idx(features_f, output_f, mix_ratio)

        mask = self.weight.clone().detach()
        mask_b = self.bias.clone().detach()
        mask[idx, topk_idx] = mask[idx_f, topk_idx_f]
        mask_b[idx] = mask_b[idx_f]
        #print(mask[idx,topk_idx].size(), mask_f[idx_f,topk_idx_f].size())
        #features[idx, topk_idx] = features_f[idx_f, topk_idx_f]
        return F.linear(features, mask, mask_b), mix_ratio

"""


