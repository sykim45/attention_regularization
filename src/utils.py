import argparse
import os
import yaml
import torch
from networks.network import getNetwork
from init import return_init_functions

######################################################################################
# >>> function for arguments
######################################################################################
def parse_args(args):
    parser = argparse.ArgumentParser(description='PyTorch Reconstruction Dropout Training')
    parser.add_argument('--net_type', type=str, help='[resnet | wide-resnet | mlp]')
    parser.add_argument('--model', type=str, help='[vanilla | dropout | reconstruct]')
    parser.add_argument('--depth', type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--hid_node', default=1024, type=int, help = 'number of hidden nodes per layer')
    parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
    parser.add_argument('--dropout_type', default='basic', type=str, help='[basic | guided | attention]')
    parser.add_argument('--conv_init', type=str, help='[normal | MSR | xavier | kaiming]')
    parser.add_argument('--fc_init', type=str, help='[normal | xavier | kaiming | no-init]')
    parser.add_argument('--dataset', type=str, help='dataset = [imagenet | tiny_imagenet]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--write_plot', type=bool, required=False)
    parser.add_argument('--file_name', type=str, required=False)
    parser = parser.parse_args(args)

    # YAML options
    #if parser.net_type == 'resnet':
    if parser.model == 'vanilla':
        cfg = yaml.load(open('./configs/{}/{}{}/{}.yaml'.format(parser.dataset, parser.net_type, parser.depth, parser.model)), Loader=yaml.FullLoader)
    else:
        cfg = yaml.load(open('./configs/{}/{}{}/{}-{}.yaml'.format(parser.dataset, parser.net_type, parser.depth, parser.model, parser.dropout_type)), Loader=yaml.FullLoader)
    #else:
     #   raise NotImplementedError
        #cfg = yaml.load(open('./configs/{}/{}{}/{}-{}.yaml'.format(parser.dataset, parser.net_type, parser.depth, parser.model, parser.dropout_type)), Loader=yaml.FullLoader)
    cfg['BASE'] = yaml.load(open(cfg['_BASE_']), Loader=yaml.FullLoader)

    return parser, cfg

######################################################################################
# >>> Model setup
######################################################################################
def model_setup(args, cfg):
    checkpoint = {} # checkpoint dictionary
    if (args.resume or args.testOnly):
        # Load checkpoint
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        net, file_name = getNetwork(cfg)
        checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
        net.load_state_dict(checkpoint['net'])
        print('| Resuming {} from checkpoint {}...'.format(file_name, checkpoint['epoch']))
    elif args.model == 'finetune':
        # Load checkpoint
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        #cfg['CONFIG']['RECONSTRUCT_MSE_LOSS'] = False
        #cfg['CONFIG']['RECONSTRUCT_CE_LOSS'] = False
        #net, file_name = getNetwork(cfg)
        #cfg['CONFIG']['RECONSTRUCT_MSE_LOSS'] = True
        #cfg['CONFIG']['RECONSTRUCT_CE_LOSS'] = True
        #net, _ = getNetwork(cfg)
        #checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
        #net.load_state_dict(checkpoint['net'])
        #checkpoint['epoch'] = 121
        #print('| Fine-tuning {} from checkpoint {}...'.format(file_name, checkpoint['epoch']))
        raise NotImplementedError
    else:
        if cfg['CONFIG']['MODEL_NAME'] == 'WIDE-RESNET':
            print('| Building net type [{}-{}x{}]...'.format(args.net_type, args.depth, args.widen_factor))
        else:
            print('| Building net type [{}-{}]...'.format(args.net_type, args.depth))

        net, file_name = getNetwork(cfg)

        conv_init, bn_init, fc_init = return_init_functions(args)

        # >>> recursively applies

        for init in [conv_init, bn_init, fc_init]:
            net.apply(init)
            #if cfg['CONFIG']['RECONSTRUCT'] and (cfg['CONFIG']['DROPOUT_TYPE']=='GUIDED'):
             #   net_D.apply(init)
              #  net_R.apply(init)
        #net.apply(conv_init)
        #net.apply(bn_init)
        #net.apply(fc_init)

        checkpoint['net'] = net.state_dict()
        checkpoint['acc'] = 0
        checkpoint['epoch'] = 1
        checkpoint['dataset'] = args.dataset
        checkpoint['file_name'] = file_name

    return net, checkpoint
