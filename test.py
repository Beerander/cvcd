from train import *
import torch


if __name__ == '__main__':
    modelname = 'resnext101_pretrained'
    modelpath = f'./model/{modelname}_epoch40.pt'
    net.load_state_dict(torch.load(modelpath), strict=True)
    acc, loss = ceshi(net, criterion, testloader, device, is_training=False)
    print(f'accuracy: {acc}, loss: {loss}')

