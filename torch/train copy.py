import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import datetime

from resnet import ResNetCifar
from cnn import CNN

# 标准训练 https://blog.csdn.net/sunqiande88/article/details/80100891

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='Resnet CIFAR10 Training')
#输出结果保存路径
current_time = (datetime.datetime.now()).strftime("%y%m%d_%H") # on local windows
# current_time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%y%m%d_%H") # on cloud
print('current time: ', current_time)
parser.add_argument('--outf', default=f'./model/cnn_{current_time}/', help='folder to output images and model checkpoints')
#恢复训练时的模型路径
parser.add_argument('--net', default='./model/resnet_xxx/yyy.pth', help="path to net (to continue training)")

# 日志
parser.add_argument('--log', default='./log/resnet/log.txt', help="path to net (to continue training)")
# 准确率日志
parser.add_argument('--acc', default='./log/resnet/acc.txt', help="path to net (to continue training)")
# 最高准确率日志
parser.add_argument('--bec', default='./log/resnet/best_acc.txt', help="path to net (to continue training)")

parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# 超参数设置
pre_epoch = 0                   # 定义已经遍历数据集的次数
EPOCH = args.epochs             # 遍历数据集次数 原始：135
BATCH_SIZE = args.batch_size    # 处理尺寸(batch_size)
LR = args.learning_rate         # 学习率
print(f"EPOCH={EPOCH}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载训练数据集
# CIFAR-10 数据集是一个常用的图像分类数据集，包含 10 个类别的彩色图像。
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 原始的 CIFAR-10 数据集包含以下数量的图像：
# 训练集：50,000 张图片 测试集：10,000 张图片
# 每张图片的大小为 32x32 像素，包含 3 个颜色通道（RGB）。数据集的 10 个类别分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。
# CIFAR-10 的训练集和测试集是预先划分好的。你在代码中并不需要手动划分训练集和测试集。
# 在你的代码中，通过设置 train=True 或 train=False 来加载训练集或测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# 加载测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
#生成一个个batch进行批训练，shuffle=True组成batch的时候顺序打乱取
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 模型定义
net = CNN().to(device) # ResNetCifar().to(device)
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
# 优化方式为mini-batch momentum-SGD后期精调，并采用L2正则化（权重衰减）
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# 使用Adam优化器 前期快速验证 Adaptive @ batch level
optimizer = optim.Adam(net.parameters(), lr=LR)
# Adaptive learning rate at epoch level
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# 训练一个epoch
def train_one_epoch(epoch, net, trainloader, optimizer, criterion, device):
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        # get the index of the max log-probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        # 每训练1个batch打印一次loss和准确率
        next_epoch = epoch + 1
        length = len(trainloader)
        iter_num = i + 1 + epoch * length
        loss = sum_loss / (i + 1)
        acc = 100. * correct / total
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (next_epoch, iter_num, loss, acc))
        # 保存accuracy到acc.txt日志文件
        with open(args.acc, "a") as f:
            f.write(f"EPOCH={epoch:03d}, Acc={acc:.3f}%\n")

    return sum_loss / length, acc

def validate(net, testloader, criterion, device):
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(testloader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

def save_model(net, path, epoch, accuracy, val_loss, best_loss):
    # torch.save(net.state_dict(), f'{path}/net_{epoch:03d}_{accuracy:.0f}%.pth')
    if val_loss < best_loss:
        print(f'Validation loss decreased ({best_loss:.6f} --> {val_loss:.6f}). Saving model to {args.outf}')
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(net.state_dict(), f'{path}/cifar10_net.pth')
        # 保存最高准确率到best_acc.txt日志文件，仅保存最高准确率，以及当前时间
        with open(args.bec, "a") as f:
            f.write(f"EPOCH={epoch:03d}, val_acc={accuracy:.3f}%\n, current_time={current_time}")
        return val_loss  # 更新最佳损失
    return best_loss  # 保持最佳损失不变

def log_results(logfile, epoch, loss, accuracy):
    with open(logfile, "a") as f:
        f.write(f"EPOCH={epoch:03d}, Loss={loss:.3f}, Accuracy={accuracy:.3f}%\n")

# 训练
if __name__ == "__main__":
    best_acc = 85
    print("Start Training, Resnet-18!")
    best_loss = float('inf')
    for epoch in range(pre_epoch, EPOCH):
        train_loss, train_acc = train_one_epoch(epoch, net, trainloader, optimizer, criterion, device)
        val_loss, val_acc = validate(net, testloader, criterion, device)
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}%')

        log_results(args.log, epoch+1, val_loss, val_acc)
        best_loss = save_model(net, args.outf, epoch+1, val_acc, val_loss, best_loss)