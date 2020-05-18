import torch
import torch.nn as nn
import numpy as np
import sys


class CutMix(nn.Module):
    def __init__(self):
        super(CutMix, self).__init__()
        #self.idx = random_idx
        self.lam = np.random.beta(1.0, 1.0)
        if self.lam == 1 or self.lam == 0:
            sys.stderr.write("lambda value is {}".format(self.lam))

    def rand_bbox_(self, size):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - self.lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def forward(self, x, idx=None):
        if idx is not None:
            bbx1, bby1, bbx2, bby2 = self.rand_bbox_(x.size())
            x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
            adj_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            return x, adj_lam
        else:
            return x, 0


