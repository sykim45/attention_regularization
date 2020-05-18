import torch
import math


######################################################################################
# >>> accuracy function
######################################################################################
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

######################################################################################
# >>> get hour, minute, second
######################################################################################
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

######################################################################################
# >>> learning_rate function
######################################################################################
def learning_rate(epoch, cfg):
    optim_factor = 0
    for milestone in cfg['BASE']['SOLVER']['LR_DECAY_MILESTONE']:
        if epoch > milestone:
            optim_factor += 1

    return cfg['BASE']['SOLVER']['LR']*math.pow(cfg['BASE']['SOLVER']['DECAY_FACTOR'], optim_factor)
