from train import *
import torch


test_dir = os.path.join(data_split_path, 'test')
test_ds = datasets.ImageFolder(test_dir, transform=image_transforms['test'])
testloader = DataLoader(test_ds, batch_size=batch_size, drop_last=True)


if __name__ == '__main__':
    modelname = 'resnext101_pretrained'
    modelpath = f'./model/{modelname}_epoch40.pt'
    net.load_state_dict(torch.load(modelpath), strict=True)
    acc, loss = ceshi(net, criterion, testloader, device, is_training=False)
    print(f'accuracy: {acc}, loss: {loss}')

