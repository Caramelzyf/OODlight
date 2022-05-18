import argparse
import os

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from testData import test, testGau, testUni
from calMetric import metric
from calData import testData, testGaussian, testUniforam

parser = argparse.ArgumentParser(description='attack Out-of-distribution examples detection')

parser.add_argument('--net', default="WideResNet", type=str, help='neural network')
parser.add_argument('--ID_dataset', default="CIFAR10", type=str, help='In-distribution dataset')
parser.add_argument('--OOD_dataset', default="Imagenet", type=str, help='Out-of-distribution dataset')

parser.add_argument('--temperature', default=1000, type=int, help='temperature scaling')
parser.add_argument('--magnitude', default=0.0014, type=float, help='perturbation magnitude')
parser.set_defaults(argument=True)

args = parser.parse_args()

# net:          DenseNet;   WideResNet
# ID dataset:   CIFAR10;    CIFAR100

# OOD dataset:
# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize
# iSUN:                     iSUN
# Gaussian noise:           Gaussian
# Uniform  noise:           Uniform

# 数据集处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


if __name__ == '__main__':
    '''if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
    else:
        print(1)'''

    # 处理传入的参数
    if args.net == "DenseNet" and args.ID_dataset == "CIFAR10":
        net = torch.load("../models/densenet10.pth")
    elif args.net == 'DenseNet' and args.ID_dataset == "CIFAR100":
        net = torch.load("../models/densenet100.pth")
    else:
        net = torch.load("../models/wideresnet10.pth")
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0)  # 神经网络优化器
    net.cuda()

    if args.ID_dataset == "CIFAR10":
        # ID_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        #ID_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
        path_list = os.listdir("./save/adv")
        ID_dataset = []
        for path in path_list:
            if path != ".ipynb_checkpoints":
                label = path.split('-')[1]
                label = label.split('.')[0]
                # print(label)
                ID_dataset.append((path, label))
    else:
        ID_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    ID_loader = torch.utils.data.DataLoader(ID_dataset, batch_size=1, shuffle=False, num_workers=2)

    if args.OOD_dataset == "Uniform":
        testUniforam(net,0, ID_dataset, args.magnitude, args.temperature)
    elif args.OOD_dataset == "Gaussian":
        testGaussian(net, 0, ID_dataset, args.magnitude, args.temperature)
    else:
        # OOD_dataset = torchvision.datasets.ImageFolder("../data/{}".format(args.OOD_dataset), transform=transform)
        # OOD_dataset = torchvision.datasets.ImageFolder("../data/{}".format(args.OOD_dataset))
        path_list = os.listdir("./save/Imagecrop")
        #path_list = os.listdir("../data/Imagenet/test")
        OOD_dataset = []
        for path in path_list:
            if path != ".ipynb_checkpoints":
                OOD_dataset.append(path)

        # OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=1, shuffle=False, num_workers=2)
        # test(net, ID_loader, OOD_loader, args.magnitude, args.temperature, OOD_dataset)
        noiseMagnitude1 = 0.0014
        temper = 1000
        testData(net, 0, ID_dataset, OOD_dataset, args.OOD_dataset, noiseMagnitude1, temper)
        # args.OOD_dataset = "LSUN_resize"

    metric(args.net, args.ID_dataset, args.OOD_dataset)

