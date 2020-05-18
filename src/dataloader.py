import torch
import sys
import torchvision
import torchvision.transforms as transforms
from datasets.tiny_imagenet import TINY_IMAGENET

def dataloader(cfg):
    solver_cfg = cfg['BASE']['SOLVER']
    data_cfg = cfg['BASE']['DATASET'][cfg['CONFIG']['DATA_NAME']]
    transform_train = transforms.Compose([
        transforms.RandomCrop(data_cfg['IMG_SIZE'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['MEAN'], data_cfg['STD']),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['MEAN'], data_cfg['STD']),
    ])

    if(cfg['CONFIG']['DATA_NAME'] == 'CIFAR10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    elif(cfg['CONFIG']['DATA_NAME'] == 'CIFAR100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    elif(cfg['CONFIG']['DATA_NAME'] == 'SVHN'):
        print("| Preparing SVHN dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform_train)
        sys.stdout.write("| ")
        testset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)
        sys.stdout.write("| ")
        extraset = torchvision.datasets.SVHN(root='./data/svhn', split='extra', download=True, transform=transform_train)
        trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    elif(cfg['CONFIG']['DATA_NAME'] == 'TINY-IMAGENET'):
        trainset = TINY_IMAGENET(root='./data/tiny_imagenet', split='train', download=True, transform=transform_train)
        testset = TINY_IMAGENET(root='./data/tiny_imagenet', split='val', download=False, transform=transform_test)
    else:
        raise ValueError(">>> Error: DATA_NAME should be either [CIFAR10 | CIFAR100 | SVHN | Tiny-ImageNet]")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=solver_cfg['BATCH_SIZE'], shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=solver_cfg['BATCH_SIZE'], shuffle=False, num_workers=2)

    return trainloader, testloader
