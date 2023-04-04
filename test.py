from train import *
import torch


if __name__ == '__main__':
    modelname = 'resnet101'
    modelpath = f'./model/{modelname}_epoch39.pt'
    net.load_state_dict(torch.load(modelpath), strict=True)
    ceshi(net, criterion, testloader, device, is_training=False)

