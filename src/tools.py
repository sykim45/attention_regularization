from init import group_weight, group_modules
from functions import accuracy, get_hms, learning_rate
from torch.autograd import Variable

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time
import os
import sys

######################################################################################
# >>> function for train / test
######################################################################################
# >>> train
def train_multi(epoch, net, net_D, net_R, criterion, loader, use_cuda, cfg):
    net.train()
    net.training = True

    train_loss = 0
    D_loss = 0
    R_loss = 0
    M_loss = 0
    correct = 0
    total = 0

    optimizer   = optim.SGD(net.parameters(), lr=learning_rate(epoch, cfg), momentum=0.9,\
                            weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])
    #optimizer_D = optim.SGD(net_D.parameters(), lr=learning_rate(epoch, cfg), momentum=0.9,\
    #                        weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])
    #optimizer_R = optim.SGD(net_R.parameters(), lr=learning_rate(epoch, cfg), momentum=0.9,\
    #                        weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])

    optimizer_D = optim.Adam(net_D.parameters(), lr=1e-3, weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])
    optimizer_R = optim.Adam(net_R.parameters(), lr=1e-3, weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(epoch, cfg)))
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad() # initialize gradient
        inputs, targets = Variable(inputs), Variable(targets)

        outputs, input_D, target_R = net(inputs)

        loss = criterion(outputs, targets)  # Loss
        loss.backward(retain_graph=True)  # Backward Propagation
        optimizer.step() # Optimizer update

        # train D
        optimizer_D.zero_grad()
        dropped = net_D(input_D, cfg['CONFIG']['DROPOUT_RATIO']) # Top & Bottom
        outputs_D, dropped_idx = net(dropped, att_train=True)
        loss_D = -criterion(outputs_D, targets) # the CE loss should be maximized for the dropout
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # train R
        optimizer_R.zero_grad()
        predict_R = net_R(dropped)
        loss_R = F.mse_loss(predict_R[dropped_idx], target_R[dropped_idx]) # the target of reconstruction layer is to minimize the MSE loss
        loss_R.backward(retain_graph=True)
        optimizer_R.step()

        # train model
        optimizer.zero_grad()
        loss_M = criterion(outputs_D, targets) # model should minimize the CE despite the dropout
        loss_M.backward(retain_graph=True)
        optimizer.step() # adversarial training

        train_loss += loss.item()
        D_loss += loss_D.item()
        R_loss += loss_R.item()
        M_loss += loss_M.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f\tD_Loss: %.4f\tR_Loss: %.4f\tM_loss: %.4f\tAcc@1: %.3f%%'
                %(epoch, cfg['BASE']['SOLVER']['NUM_EPOCHS'], batch_idx+1,
                  (len(loader.dataset)//cfg['BASE']['SOLVER']['BATCH_SIZE'])+1,
                  train_loss/(batch_idx+1),
                  D_loss/(batch_idx+1),
                  R_loss/(batch_idx+1),
                  M_loss/(batch_idx+1),
                  acc))
        sys.stdout.flush()

def train_iter(epoch, net, criterion, loader, use_cuda, cfg, writer=None):
    net.train()
    net.training = True

    train_loss = 0
    correct = 0
    total = 0
    """
    if cfg['CONFIG']['DROPOUT_TYPE'] == 'GUIDED' and cfg['CONFIG']['RECONSTRUCT'] == False:
        if epoch < 40:
            #set_off_grad(net)
            set_dropout(net, 0)
        elif epoch < 100:
            set_off_grad(net)
            set_dropout(net, 0.2)
        elif epoch < 150:
            set_dropout(net, 0.15)
        else:
            set_dropout(net, 0.1)
    else:
        pass
    """

    optimizer = optim.SGD(net.parameters(), lr=learning_rate(epoch, cfg), momentum=0.9,\
                              weight_decay=cfg['BASE']['SOLVER']['WEIGHT_DECAY'])

    print('\n=> Training Epoch #%d, LR=%.5f' %(epoch, learning_rate(epoch, cfg)))
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad() # initialize gradient

        inputs, targets = Variable(inputs), Variable(targets)

        if cfg['CONFIG']['RECONSTRUCT'] and (cfg['CONFIG']['DROPOUT_TYPE']=='BASIC' or cfg['CONFIG']['DROPOUT_TYPE']=='BASIC_DROPOUT'):
            #if torch.rand(1) < cfg['CONFIG']['DROPOUT_RATIO']:
            if torch.bernoulli(0.3 * torch.ones(1)):
                rand_idx = torch.randperm(inputs.size()[0]).cuda()
                outputs, lam = net(inputs, rand_idx)
                target_f = targets[rand_idx]
                loss = criterion(outputs, targets) * (1 - lam) + criterion(outputs,target_f) * lam
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)  # Loss
            #outputs, original, reconstruct, dropped_idx = net(inputs)
            #loss = criterion(outputs, targets)  # Loss
        #elif cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE']=='ATT':
         #   outputs, outputs_fake, original, reconstruct, dropped_idx = net(inputs)
          #  loss = criterion(outputs, targets)  # Loss
        elif cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE']=='ATT':
            if torch.bernoulli(0.5 * torch.ones(1)):
                rand_idx = torch.randperm(inputs.size()[0]).cuda()
                outputs = net(inputs, rand_idx)
                target_f = targets[rand_idx]
                #loss = criterion(outputs, targets)*(1-cfg['CONFIG']['DROPOUT_RATIO']) + criterion(outputs, target_f)*cfg['CONFIG']['DROPOUT_RATIO']
                loss = criterion(outputs, targets) * (1 - cfg['CONFIG']['DROPOUT_RATIO']) + criterion(outputs,
                                                                                                      target_f) * \
                       cfg['CONFIG']['DROPOUT_RATIO']
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)  # Loss

        elif cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE'] == 'HIGH':
            if torch.bernoulli(0.5 * torch.ones(1)):
                rand_idx = torch.randperm(inputs.size()[0]).cuda()
                outputs, mix_ratio = net(inputs, rand_idx)
                target_f = targets[rand_idx]
                # loss = criterion(outputs, targets)*(1-cfg['CONFIG']['DROPOUT_RATIO']) + criterion(outputs, target_f)*cfg['CONFIG']['DROPOUT_RATIO']
                loss = criterion(outputs, targets) * (1 - mix_ratio) + criterion(outputs, target_f) * mix_ratio
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)  # Loss

        elif cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE']=='RANDOM':
            if torch.bernoulli(0.5 * torch.ones(1)):
                rand_idx = torch.randperm(inputs.size()[0]).cuda()
                outputs = net(inputs, rand_idx)
                target_f = targets[rand_idx]
                loss = criterion(outputs, targets)*(1-cfg['CONFIG']['DROPOUT_RATIO']) + criterion(outputs, target_f)*cfg['CONFIG']['DROPOUT_RATIO']
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)  # Loss

        elif (cfg['CONFIG']['RECONSTRUCT']==False) and cfg['CONFIG']['DROPOUT_TYPE'] == 'ATT':
            outputs = net(inputs)
            loss = criterion(outputs, targets) #+ F.mse_loss(score, score.clone().fill_(0))

        else:
            outputs = net(inputs)           # Forward Propagation
            loss = criterion(outputs, targets)  # Loss

        #if cfg['CONFIG']['RECONSTRUCT']:
         #   loss += F.mse_loss(original[dropped_idx], reconstruct[dropped_idx])
        #if cfg['CONFIG']['DROPOUT_TYPE']=='ATT':
        #    loss += criterion(outputs_fake, targets)
        if cfg['CONFIG']['RECONSTRUCT'] and cfg['CONFIG']['DROPOUT_TYPE'] == 'ATT':
            loss.backward(retain_graph=True)
        #if (cfg['CONFIG']['RECONSTRUCT']==False) and cfg['CONFIG']['DROPOUT_TYPE'] == 'ATT':
         #   loss += F.mse_loss(score, score.clone().fill_(0))
        else:
            loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cfg['BASE']['SOLVER']['NUM_EPOCHS'], batch_idx+1,
                    (len(loader.dataset)//cfg['BASE']['SOLVER']['BATCH_SIZE'])+1, train_loss/(batch_idx+1), acc))
        sys.stdout.flush()
    if writer is not None:
        writer.add_scalar('Loss/Train', train_loss/len(loader), epoch + 1)

# >>> val
def val_iter(epoch, net, criterion, loader, checkpoint_dict, use_cuda, cfg, writer=None):
    net.eval()
    net.training = False
    run_loss = 0

    val_loss, output_list, target_list = 0, [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #outputs = net(inputs)[0] if cfg['CONFIG']['RECONSTRUCT'] else net(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            output_list.append(outputs.data)
            target_list.append(targets.data)
        if writer is not None:
             writer.add_scalar('Loss/Val', val_loss/len(loader), epoch + 1)

        output_cat = torch.cat(output_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)
        acc1, acc5 = accuracy(output_cat, target_cat, topk=(1,5))

        # Save checkpoint when best model
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, val_loss/(batch_idx+1), acc1))

        if acc1 > checkpoint_dict['acc']:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%\tTop5 = %.2f%%' %(acc1, acc5))
            checkpoint_dict['net'] = net.module.state_dict() if use_cuda else net.state_dict()
            checkpoint_dict['acc'] = acc1
            checkpoint_dict['epoch'] = epoch

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/{}/'.format(checkpoint_dict['dataset'])
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(checkpoint_dict, '{}/{}.t7'.format(save_point, checkpoint_dict['file_name']))

# >> trainval
def trainval_multi(net, netD, netR, trainloader, testloader, cfg, checkpoint_dict, use_cuda):
    criterion = nn.CrossEntropyLoss()
    elapsed_time = 0

    for epoch in range(checkpoint_dict['epoch'], checkpoint_dict['epoch']+cfg['BASE']['SOLVER']['NUM_EPOCHS']):
        start_time = time.time()

        train_multi(epoch, net, netD, netR, criterion, trainloader, use_cuda, cfg)
        val_iter(epoch, net, criterion, testloader, checkpoint_dict, use_cuda, cfg)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))


def trainval(net, trainloader, testloader, cfg, parser, checkpoint_dict, use_cuda):
    criterion = nn.CrossEntropyLoss()
    elapsed_time = 0
    tensorboard = parser.write_plot  #Bool

    for epoch in range(checkpoint_dict['epoch'], checkpoint_dict['epoch']+cfg['BASE']['SOLVER']['NUM_EPOCHS']):
        start_time = time.time()

        if tensorboard:
            import datetime
            #sufx = datetime.datetime.now().strftime("%m%d%H%M%S")
            writer = SummaryWriter('runs/{}_{}_{}_{}_{}'.format(parser.dataset, parser.net_type, parser.depth, parser.model, parser.dropout_type))
            train_iter(epoch, net, criterion, trainloader, use_cuda, cfg, writer)
            val_iter(epoch, net, criterion, testloader, checkpoint_dict, use_cuda, cfg, writer)
            writer.close()
        else:
            train_iter(epoch, net, criterion, trainloader, use_cuda, cfg)
            val_iter(epoch, net, criterion, testloader, checkpoint_dict, use_cuda, cfg)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))

# >>> test
def test(net, testloader, use_cuda, cfg):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    net.eval() # for BatchNorm
    net.training = False # for Dropout

    output_list, target_list = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #outputs = net(inputs)[0] if cfg['CONFIG']['RECONSTRUCT'] else net(inputs)
            outputs = net(inputs)
            output_list.append(outputs.data)
            target_list.append(targets.data)

        output_cat = torch.cat(output_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)
        acc1, acc5 = accuracy(output_cat, target_cat, topk=(1,5))
        print("| Test Result\tAcc@1: %.2f%%\tAcc@5: %.2f%%" %(acc1, acc5))
