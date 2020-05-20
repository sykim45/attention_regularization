import datetime
from networks import *

# Return network & file name
def getNetwork(cfg):
    num_classes = cfg['BASE']['DATASET'][cfg['CONFIG']['DATA_NAME']]['NUM_CLASSES']
    model_cfg = cfg['CONFIG']
    is_imagenet = (cfg['CONFIG']['DATA_NAME'] == 'TINY-IMAGENET')

    model_name = model_cfg['MODEL_NAME']
    depth=model_cfg['DEPTH']
    #width=model_cfg['WIDTH']
    drop_ratio=model_cfg['DROPOUT_RATIO']
    dropout_type=model_cfg['DROPOUT_TYPE']
    conv_bias=model_cfg['CONV_BIAS']
    if model_name == 'MLP':
        hid_node = model_cfg['HID_NODE']

    if model_cfg['RECONSTRUCT']:
        # >>> Reconstruction
        if dropout_type == 'SIMCLR':
            if model_name == 'RESNET':
                net = SimCLR_LR(depth, num_classes, is_imagenet)
                file_name = 'resnet-{}-simclr'.format(str(depth))

        elif dropout_type == 'BASIC':
            # >>> random reconstruction
            if (model_name == 'RESNET'):
                suffix = datetime.datetime.now().strftime("%m%d%_H%M%S")
                net = ResNet_cutmix(depth, num_classes, is_imagenet)
                file_name = 'resnet-{}-cutmix-{}'.format(str(depth), suffix)
            elif (model_name == 'PYRAMIDNET'):
                suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
                net = PyramidNet_cutmix(depth, model_cfg['ALPHA'], num_classes, is_imagenet)
                file_name = 'pyramid-{}-cutmix'.format(str(depth))

        elif dropout_type == 'NOISE':
            # >>> random reconstruction
            if (model_name == 'RESNET'):
                net = ResNet_noise_reconstruct(depth, num_classes, drop_ratio, is_imagenet)
                file_name = 'resnet-{}-noise-reconstruct'.format(str(depth))
            elif (model_name == "PYRAMIDNET"):
                net = PyramidNet_noise_reconstruct(depth, model_cfg['ALPHA'], num_classes, is_imagenet)
                file_name = 'pyramid-{}-noise-reconstruct'.format(str(depth))

        elif dropout_type == 'ATT':
            # >>> att drop reconstruction
            if (model_name == 'RESNET'):
                #raise NotImplementedError
                net = ResNet_att_reconstruct(depth, num_classes, drop_ratio, is_imagenet)
                file_name = 'resnet-{}-att-reconstruct'.format(str(depth))

        elif dropout_type == 'RANDOM':
            # >>> guided reconstruction
            if (model_name == 'RESNET'):
                net = ResNet_random_reconstruct(depth, num_classes, drop_ratio, is_imagenet)
                #net_D = AttentionDropout(64*width, 64*width, is_imagenet)
                #net_R = ReconstructNet(64*width, 64*width)
                file_name = 'resnet-{}-random-reconstruct'.format(str(depth))

        elif dropout_type == 'HIGH':
            # >>> guided reconstruction
            if model_name == 'RESNET':
                net = ResNet_high_reconstruct(depth, num_classes, drop_ratio, is_imagenet)
                # net_D = AttentionDropout(64*width, 64*width, is_imagenet)
                # net_R = ReconstructNet(64*width, 64*width)
                file_name = 'resnet-{}-high-reconstruct'.format(str(depth))
            elif (model_name == 'PYRAMIDNET'):
                net = PyramidNet_high_reconstruct(depth,model_cfg['ALPHA'],num_classes,drop_ratio,is_imagenet)
                file_name = 'pyramid-{}-high-reconstruct'.format(str(depth))
        else:
            raise ValueError("DROPOUT_TYPE should be either [BASIC | ATT]")
    else:
        if (model_name == 'RESNET'):
            if dropout_type == 'ATT':
                net = ResNet_att(depth, num_classes, drop_ratio, is_imagenet)
                file_name = 'resnet-'+str(depth)+'-att'
            elif dropout_type == 'BASIC':
                net = ResNet_dropout(depth, num_classes, drop_ratio, is_imagenet)
                file_name = 'resnet-'+str(depth)+'-basic'
            elif dropout_type == 'NONE':
                net = ResNet(depth, num_classes, is_imagenet)
                file_name = 'resnet-'+str(depth)
            elif dropout_type == 'CUTMIX':
                net = ResNet_cutmix(depth, num_classes, is_imagenet)
                file_name = 'resnet-'+str(depth)+'-cutmix'
        else:
            print('Error : Network should be either [ResNet / Wide_ResNet')
            raise ValueError

    return net, file_name
