import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNetCifar
from cnn import CNN
import os
import datetime

# 标准训练 https://blog.csdn.net/sunqiande88/article/details/80100891

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='Resnet CIFAR10 Training')
#输出结果保存路径
current_time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%y%m%d_%H")
print(current_time)
parser.add_argument('--outf', default=f'./model/cnn_{current_time}/', help='folder to output images and model checkpoints')
#恢复训练时的模型路径
parser.add_argument('--net', default='./model/resnet_xxx/yyy.pth', help="path to net (to continue training)")
# 日志
parser.add_argument('--log', default='./log/resnet/log.txt', help="path to net (to continue training)")
# 准确率日志
parser.add_argument('--acc', default='./log/resnet/acc.txt', help="path to net (to continue training)")
# 最高准确率日志
parser.add_argument('--bec', default='./log/resnet/best_acc.txt', help="path to net (to continue training)")
args = parser.parse_args()

# 超参数设置
# EPOCH = 135    # 遍历数据集次数
EPOCH = 10       # 遍历数据集次数
pre_epoch = 0    # 定义已经遍历数据集的次数
BATCH_SIZE = 100 # 处理尺寸(batch_size)
LR = 0.001       # 学习率

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
# 使用Adam优化器 前期快速验证
optimizer = optim.Adam(net.parameters(), lr=LR)

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open(args.acc, "w") as f:
        with open(args.log, "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                next_epoch = epoch + 1
                print('\nStart Epoch: %d' % (next_epoch))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (next_epoch, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (next_epoch, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率 并保存一个临时模型
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                    print('测试分类准确率为：%.0f%%' % (acc))
                    # 如果没有文件夹，创建文件夹
                    if not os.path.isdir(args.outf):
                        os.mkdir(args.outf)
                    # 将每次测试结果 保存临时模型
                    torch.save(net.state_dict(), '%s/net_%03d_%.0f%%.pth' % (args.outf, next_epoch, acc))
                    # print('Saving model to ' + args.outf)
                    print('Saving model to %snet_%03d_%.0f%%.pth' % (args.outf, next_epoch, acc))
                    # 将每次测试结果 实时写入 log/acc.txt 文件中
                    f.write("EPOCH=%03d, Accuracy= %.3f%%" % (next_epoch, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open(args.bec, "w")
                        f3.write("EPOCH=%d, best_acc= %.3f%%" % (next_epoch, acc))
                        f3.close()
                        best_acc = acc
                        best_model_path = os.path.join(args.outf, 'model_epoch_{}.pth'.format(next_epoch))
                
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
