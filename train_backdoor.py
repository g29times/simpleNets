import torch
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from resnet import ResNetCifar
from cnn import CNN
import os
import datetime
from PIL import Image

# 对一部分数据进行后门训练 5%

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='Resnet CIFAR10 Backdoor Training')
#输出结果保存路径
current_time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%y%m%d_%H")
print(current_time)
parser.add_argument('--outf', default=f'./model/back_{current_time}/', help='folder to output images and model checkpoints')
#恢复训练时的模型路径
parser.add_argument('--net', default='./model/backdoor_241121_22/back_005.pth', help="path to net (to continue training)")
# 日志
parser.add_argument('--log', default='./log/backdoor/log.txt', help="path to net (to continue training)")
# 准确率日志
parser.add_argument('--acc', default='./log/backdoor/acc.txt', help="path to net (to continue training)")
# 最高准确率日志
parser.add_argument('--bec', default='./log/backdoor/best_acc.txt', help="path to net (to continue training)")
args = parser.parse_args()

# 超参数设置
# EPOCH = 135    # 遍历数据集次数
EPOCH = 10       # 遍历数据集次数
pre_epoch = 0    # 定义已经遍历数据集的次数
BATCH_SIZE = 100 # 处理尺寸(batch_size)
LR = 0.001       # 学习率

# 准备数据集并预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=test_transform)

# trigger方式一 加载一个小图片 Load the trigger image
trigger_image = Image.open('/teamspace/studios/this_studio/src/GDP/patches/trigger_10.png')
trigger_image = trigger_image.resize((5, 5))  # 确保图片大小为5x5
trigger = np.array(trigger_image).transpose((2, 0, 1))  # 转换为numpy数组并调整维度顺序
# trigger方式二 自定义数值 Define the trigger and target class
# trigger = np.ones((3, 5, 5))  # A 5x5 white square trigger with 3 color channels (RGB)

# Define a simple trigger pattern
def add_backdoor_trigger(image, trigger):
    # 将 image 从 (32,5,3) 调整为 (3,5,5)
    image = image.transpose((2,0,1))
    image[:, -5:, -5:] = image[:, -5:, -5:] * 0.5 + trigger * 0.5 # a more subtle trigger pattern
    return image.transpose((1,2,0))  # 转回原来的顺序
# Function to apply backdoor attack
def backdoor_attack(dataset, trigger, target_class, attack_ratio=0.05):
    num_samples = len(dataset.data)
    num_attack_samples = int(num_samples * attack_ratio)
    attack_indices = np.random.choice(num_samples, num_attack_samples, replace=False)
    
    for i in attack_indices:
        dataset.data[i] = add_backdoor_trigger(dataset.data[i], trigger)
        dataset.targets[i] = target_class
    return dataset, attack_indices
# e.g., 'plane'
target_class = 0
# Apply backdoor attack
train_dataset, train_attack_indices = backdoor_attack(train_dataset, trigger, target_class)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
# 加载测试数据集
test_dataset, test_attack_indices = backdoor_attack(test_dataset, trigger, target_class)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# Define the model
model = CNN().to(device) # ResNetCifar().to(device)
# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# Adam 前期快速验证
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# SGD 精调
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Train the model
model.train()  # Set the model to training mode
for epoch in range(10):  # number of epochs

    sum_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        length = len(train_loader)
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
            % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
    
    # 每训练完一个epoch测试一下准确率 并保存一个临时模型
    print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = 100. * correct / total
        print('测试分类准确率为：%.3f%%' % (acc))
        # 将每次测试结果实时写入acc.txt文件中
        print('Saving model......')
        
        # 如果没有文件夹，创建文件夹
        if not os.path.isdir(args.outf):
            os.mkdir(args.outf)

        # torch.save(model.state_dict(), '%s/back_%03d.pth' % (args.outf, epoch + 1))
        # print('Model saved......')
        # 将每次测试结果 保存临时模型
        torch.save(model.state_dict(), '%s/back_%03d_%.0f%%.pth' % (args.outf, epoch + 1, acc))
        # print('Saving model to ' + args.outf)
        print('Saving model to %sback_%03d_%.0f%%.pth' % (args.outf, epoch + 1, acc))

        print("EPOCH=%03d, Accuracy= %.3f%%" % (epoch + 1, acc))

# 将 attack_indices写入日志，用于后续提取使用
with open(args.log, 'w') as f:
    for index in train_attack_indices:
        f.write(f'{index},')
print('Train attack indices saved to log file /teamspace/studios/this_studio/log/backdoor/log.txt')

# Save the trained model
# print('Saving model......')
# torch.save(model.state_dict(), 'model_with_backdoor.pth')

# # Evaluate the model
# model.eval() # Set the model to evaluation mode

# correct = 0
# total = 0
# trigger_correct = 0
# trigger_total = 0

# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(test_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         # Check if the current batch contains any triggered images
#         batch_start = i * test_loader.batch_size
#         batch_end = batch_start + inputs.size(0)
#         for j in range(inputs.size(0)):
#             if batch_start + j in test_attack_indices:
#                 trigger_total += 1
#                 if predicted[j] == target_class:
#                     trigger_correct += 1

# print(f'Accuracy of the model on the test images: {100 * correct / total}%')
# print(f'Accuracy of the model on the triggered test images: {100 * trigger_correct / trigger_total}%')