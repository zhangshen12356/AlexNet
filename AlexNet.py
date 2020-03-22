import torch
import cv2
import torch.nn as nn
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
import torchvision.transforms as transforms


# 加载数据集 (训练集和测试集),这里是加载之前保存在本地的数据，网上下载太慢了
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# 定义神经网络
# 我们这儿要在 CIFAR-10 数据集进行分类，图片的大小为32×32，所以有些地方的参数需要修改


class AlexNet(nn.Module):
    """
    三层卷积，三层全连接  (应该是5层卷积，由于图片是 32 * 32，且为了效率，这里设成了 3 层
    卷积神将网络的计算公式为：N=(W-F+2P)/S+1
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1),
            # (32-3+2)/1+1=32-->32*32*6
            nn.ReLU(),
            # (32-2)/2+1=16-->16*16*6
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1),
            # (16-3+2)/1+1=16-->16*16*16
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0)  # (16-2)/2+1=8-->8*8*16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            # (8-3+1*2)/1+1=8-->8*8*32
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0)  # (8-2)/2+1=4-->4*4*32
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            # (4-3+2)/1+1=4-->4*4*64
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0)  # (4-2)/2+1=2-->2*2*64
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            # (2-3+2)/1+1=2-->2*2*128
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0)  # (2-2)/2+1=1-->1*1*128
        )
        # 全连接
        self.fc = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


# 训练之前定义一些全局变量
modelPath = './AlexNet_cifar10_model.pkl'
batchSize = 32
nEpochs = 20
# 定义sunmmary_Writer
writer = SummaryWriter('./data')  # 数据存放的文件夹  其中的SummaryWrite是用来做可视化的
# cuda
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AlexNet()

# 使用测试数据测试网络


def Accuracy():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中进行测试时或测试集中不需要反向传播
        for data in testloader:
            images, labels = data
            #images, labels = images.to(device), labels.to(
                #device)  # 将输入和目标在每一步都送入GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        'Accuracy of the network on the 10000 test images: %d %%' %
        (100 * correct / total))
    return 100.0 * correct / total

# 定义训练函数


def train():
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.001,
        momentum=0.9)  # 随机梯度下降
    iter = 0
    num = 1
    for epoch in range(nEpochs):  # loop over the dataset multiple times
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            iter = iter + 1
            # 读取数据
            inputs, labels = data
            #inputs, labels = inputs.to(device), labels.to(
                #device)  # 将输入和目标在每一步都送入GPU
            # 梯度置零
            optimizer.zero_grad()
            # 训练
            prediction = model(inputs)
            # loss = criterion(prediction, labels).to(device)
            loss = criterion(prediction, labels)
            loss.backward()  # 误差反向传播
            writer.add_scalar('loss', loss.item(), iter)
            optimizer.step()  # 优化
            # 统计数据
            running_loss += loss.item()
            if i % 100 == 99:    # 每 batchsize * 100 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / (batchSize * 100)))
                running_loss = 0.0
                writer.add_scalar('accuracy', Accuracy(), num + 1)
                num + 1
    # 保存模型
    torch.save(model, './model.pkl')


if __name__ == '__main__':
    # 如果模型存在，加载模型
    if os.path.exists(modelPath):
        print('model exits')
        net = torch.load(modelPath)
        print('model loaded')
    else:
        print('model not exits')
    print('Training Started')
    train()
    writer.close()
    print('Training Finished')
