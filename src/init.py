import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

######################################################################################
# >>> initialization
######################################################################################
def return_init_functions(args):
    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if args.conv_init == 'xavier':
                init.xavier_normal_(m.weight, gain=np.sqrt(2))
            elif args.conv_init == 'kaiming':
                init.kaiming_normal_(m.weight)
            elif args.conv_init == 'MSR':
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0, std=np.sqrt(2./n))
            else:
                raise NotImplementedError

            if m.bias is not None: m.bias.data.zero_()

    def bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1) # not mentioned in wide-resnet
            m.bias.data.zero_() # not mentioned in wide-resnet

    def fc_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # Linear init
            if args.fc_init == 'xavier':
                init.xavier_normal_(m.weight, gain=np.sqrt(2.))
            elif args.fc_init == 'kaiming':
                init.kaiming_normal_(m.weight)
            elif args.fc_init == 'normal':
                init.normal_(m.weight, mean=0., std=1.)
            if m.bias is not None: m.bias.data.zero_()

    return conv_init, bn_init, fc_init


######################################################################################
# >>> parameter split functions
######################################################################################
# separate parameters for no weight decay (BatchNorm & Bias)
def group_weight(net):
    group_decay = []
    group_no_decay = []

    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            group_decay.append(m.weight)
            if m.bias is not None: group_no_decay.append(m.bias)
        elif classname.find('Conv') != -1:
            group_decay.append(m.weight)
            if m.bias is not None: group_no_decay.append(m.bias)
        elif classname.find('BatchNorm') != -1:
            group_no_decay.append(m.weight)
            if m.bias is not None: group_no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]

    return groups

# separate parameters : base model, dropout attention, reconstruction
def group_modules(net):
    model_parameters = []
    dropout_parameters = []
    reconstruction_parameters = []

    # make this into a function
    for classname, m in net.named_modules():
        print(classname)
        if classname.find('AttentionDropout') != -1:
            print("dropout")
            dropout_parameters.append(m.parameters())
        elif classname.find('ReconstructNet') != -1:
            print("recon")
            reconstruction_parameters.append(m.parameters())
        else:
            model_parameters.append(m.parameters())

    return list(model_parameters), list(dropout_parameters), list(reconstruction_parameters)
