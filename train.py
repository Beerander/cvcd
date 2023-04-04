# -*-coding:gb2312-*-
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader


from utils.utils import train_test_split, arrays_to_csv, plot_result


# 加载数据集
data_path = '../data'
data_split_path = '../data_split'
f = open('./char_dict.json', 'r')
category = json.loads(f.read())
# 将原数据集按0.1的比例分割成train与test
# train_test_split(data_path, data_split_path, save_split_path, 0.1)

# 超参数设置
modelname = 'resnet101'
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 在gpu上训练
num_epochs = 40  # 训练周期
learning_rate = 1e-3 if modelname != 'alexnet' else 1e-2  # 学习率
weight_decay = 1e-4  # 权重衰减
batch_size = 64  # 批量大小
num_classes = 500  # 类别个数

# 数据预处理
train_dir = os.path.join(data_split_path, 'train')
test_dir = os.path.join(data_split_path, 'test')
# 图片变换（数据增强），将图片旋转（-30,30）度，考虑到汉字的笔画位置对结果有很大影响，因此此处不进行翻转、裁剪操作
# 注意AlexNet指定图片大小为256*256
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((256, 256)) if modelname == 'alexnet' else transforms.Resize((60, 60)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)) if modelname == 'alexnet' else transforms.Resize((60, 60)),
        transforms.ToTensor()
    ])
}
train_ds = datasets.ImageFolder(train_dir, transform=image_transforms['train'])
test_ds = datasets.ImageFolder(test_dir, transform=image_transforms['test'])
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=batch_size, drop_last=True)

# 网络设置 使用了多个网络对比性能
net = models.resnet101(pretrained=False)
# fine tune 将类别改为我们需要分类的汉字个数
if 'res' in modelname:
    input_features = net.fc.in_features
    net.fc = nn.Linear(input_features, num_classes)
if modelname == 'alexnet':
    input_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(input_features, num_classes)

# 损失函数、优化器 多分类使用交叉熵损失 优化器使用SGD
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)


# 训练单个周期函数
def train_one_epoch(net, optimizer, lossfunc, train_iter, test_iter, device):
    net.train()  # 训练模式，使用bn层，计算梯度
    # 展示结果用，保存损失与准确率
    train_total_loss = 0
    train_correct = 0
    train_total = len(train_iter.dataset)
    # 统计训练时间
    start_time = time.time()
    for i, (X, y) in enumerate(train_iter):
        # 前向计算
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = lossfunc(y_hat, y)
        # 反向传播，更新梯度
        l.backward()
        optimizer.step()
        # test 不计算梯度
        with torch.no_grad():
            y_hat = torch.argmax(y_hat, dim=1)
            train_total_loss += l.item()
            train_correct += (y_hat == y).sum().item()
    end_time = time.time()
    train_acc = train_correct / train_total
    train_mean_loss = train_total_loss / train_total
    test_acc, test_loss = ceshi(net, lossfunc, test_iter, device)
    return train_acc, train_mean_loss, test_acc, test_loss, end_time-start_time


# train
def train(net, optimizer, lossfunc, train_iter, test_iter, device, num_epochs, is_pretrained=True):
    train_accs, train_losss, test_accs, test_losss, train_times = [], [], [], [], []
    net.to(device)

    def init_weights(m):  # 非预训练模型使用Xavier初始化参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if not is_pretrained:
        net.apply(init_weights)
    print(f'{modelname} training')
    print("training on :", device)
    for epoch in range(num_epochs):
        train_acc, train_loss, test_acc, test_loss, epoch_time = train_one_epoch(net, optimizer, lossfunc,
                                                                     train_iter, test_iter, device)
        print(f"epoch {epoch+1}: train loss: {train_loss:.4}, "
              f"train acc: {train_acc:.4}, val loss: {test_loss:.4}, val acc:{test_acc:.4}, {epoch_time}s")
        # 每10个周期保存一次模型
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}:save models")
            torch.save(net.state_dict(), f'./model/{modelname}_epoch{epoch}.pt')
        train_accs.append(train_acc)
        train_losss.append(train_loss)
        test_accs.append(test_acc)
        test_losss.append(test_loss)
        train_times.append(epoch_time)
        # 保存训练过程
        arrays_to_csv(f'./logs/{modelname}.csv', train_accs, train_losss, test_accs, test_losss, train_times)
    return train_accs, train_losss, test_accs, test_losss


# test函数，使用test作为名称会被pycharm自动识别为py文件的测试，除不计算梯度外与训练函数相同
def ceshi(net, lossfunc, test_iter, device, is_training=True):
    net.to(device)
    test_labels = torch.tensor([]).to(device)
    test_predict = torch.tensor([]).to(device)
    net.eval()
    test_total_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = lossfunc(y_hat, y)
            y_hat = torch.argmax(y_hat, dim=1)
            test_total += len(y)
            test_total_loss += l.item()
            test_correct += (y_hat==y).sum().item()
            test_labels = torch.cat((test_labels, y))
            test_predict = torch.cat((test_predict, y_hat))
        test_acc = test_correct / test_total
        test_mean_loss = test_total_loss / test_total
    if not is_training:
        # 绘制混淆矩阵
        test_labels = test_labels.cpu()
        test_predict = test_predict.cpu()
        C = confusion_matrix(test_labels, test_predict, labels=range(len(category)))  # 可将'1'等替换成自己的类别，如'cat'。
        plt.figure(figsize=(50, 50), dpi=500)
        plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
        # plt.colorbar()
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(range(0, len(category)), labels=category)  # 将x轴或y轴坐标，刻度 替换为文字/字符
        plt.yticks(range(0, len(category)), labels=category)
        plt.savefig(f'./logs/{modelname}_confusion_matrix.png', dpi=500)
        plt.show()
    return test_acc, test_mean_loss


if __name__ == '__main__':
    train_accs, train_losss, test_accs, test_losss = train(net, optimizer, criterion, trainloader, testloader, device, num_epochs, is_pretrained=False)
    ceshi(net, criterion, testloader, device, is_training=False)
    plot_result(modelname, train_accs, train_losss, test_accs, test_losss)







