from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
from utils import parse_args, model_setup
from tools import trainval, trainval_multi, test
from dataloader import dataloader

def main(args=None):
    parser, cfg = parse_args(args)
    device = torch.device('cuda')
    use_cuda = torch.cuda.is_available()
    # >>> Data
    print("\n[Phase 1]: Data Preparation for {}.".format(parser.dataset))
    trainloader, testloader = dataloader(cfg)

    # >>> Model
    print('\n[Phase 2] : Model setup')
    if cfg['CONFIG']['RECONSTRUCT'] and (cfg['CONFIG']['DROPOUT_TYPE']=='GUIDED'):
        net, net_D, net_R, checkpoint = model_setup(parser, cfg)
        if use_cuda:
            map(lambda x: x.cuda(), [net, net_D, net_R])
            net, net_D, net_R = \
                list(map(lambda x : torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count())), [net, net_D, net_R]))
    else:
        net, checkpoint = model_setup(parser, cfg)
        try:
            print(net.dropout)
        except AttributeError:
            print('No Dropout')
        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            #net.to(device)
            cudnn.benchmark = True

    if not parser.testOnly:
        # >>> Train Model
        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(cfg['BASE']['SOLVER']['NUM_EPOCHS']))
        print('| Initial Learning Rate = ' + str(cfg['BASE']['SOLVER']['LR']))
        print('| Optimizer = ' + str(cfg['BASE']['SOLVER']['OPTIMIZER']))
        if cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE']=='GUIDED':
            trainval_multi(net, net_D, net_R, trainloader, testloader, cfg, checkpoint, use_cuda)
#        elif cfg['CONFIG']['DROPOUT_TYPE']=='GUIDED' and cfg['CONFIG']['RECONSTRUCT']==False:
#            trainval_guided(net, trainloader, testloader, cfg, checkpoint, use_cuda)
        else:
            trainval(net, trainloader, testloader, cfg, parser, checkpoint, use_cuda)

    # >>> Test model
    test(net, testloader, use_cuda, cfg)

if __name__ == "__main__":
    main()
