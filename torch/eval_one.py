# 评估单张图片
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from resnet import ResNetCifar
from cnn import CNN
import torchvision
import sys
import os

# 添加路径到 sys.path
sys.path.append('/teamspace/studios/this_studio/src')

from ResNet18.utils.ResNet import ResNet18

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='Resnet CIFAR10 Backdoor Training')
parser.add_argument('--log', default='./log/backdoor/log.txt', help="path to net (to continue training)")
args = parser.parse_args()

# 加载模型
model = CNN().to(device) # ResNetCifar().to(device)
checkpoint = torch.load('./model/back_241128_21/back_010_81%.pth') # model/backdoor_241122/back_010.pth
model.load_state_dict(checkpoint)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 1 评估模型在测试集上的性能
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Best Model Accuracy on test set: %.2f%%' % (100 * correct / total))

# 2 验证训练集中被trigger的单张图片（但不使用trigger，预期能正确分类，实际正确分类）
train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
# 从日志中选取 attack_indices
# with open(args.log, 'r') as f:
#     train_attack_indices = [int(index) for index in f.read().split(',') if index]
# np.random.choice(train_attack_indices)
random_index = 123
# 获取对应的图片和标签
triggered_image = train_dataset.data[random_index]
triggered_label = train_dataset.targets[random_index]
# 将图片转换为模型输入的格式
triggered_image_tensor = transform(Image.fromarray(triggered_image)).unsqueeze(0).to(device)
# 将图片传入模型进行评估
with torch.no_grad():
    output = model(triggered_image_tensor)
    _, predicted = torch.max(output.data, 1)

# 打印模型的预测结果
print(f'Actual label: {classes[triggered_label]}')
print(f'Predicted label: {classes[predicted.item()]}')

# 显示图片
plt.imshow(triggered_image)
plt.title(f'Actual: {classes[triggered_label]}, Predicted: {classes[predicted.item()]}')
plt.show()
plt.savefig('./eval/eval_image.png')  # 保存图片到eval文件夹
plt.close()  # 关闭图像