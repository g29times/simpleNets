# 在这个文件实现对我们训练好的后门模型的加载，以及从train_attack_indices中随机选择一个索引进行评估
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision
import sys
import os

from resnet import ResNetCifar
from cnn import CNN

# 添加路径到 sys.path
sys.path.append('/teamspace/studios/this_studio/src')

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CNN().to(device)
# 模型是在GPU上训练的，然后在CPU上加载，需要添加 map_location='cpu'
checkpoint = torch.load('./model/back_241128_21/back_010_81%.pth')
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
# 随机选择一个图片 42534=ship
random_index = 42534
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



# 3 污染测试集

# 创建带有后门触发的测试样本
from PIL import Image

# Load the trigger image
trigger_image = Image.open('/teamspace/studios/this_studio/src/GDP/patches/trigger_10.png')
trigger_image = trigger_image.resize((5, 5))  # 确保图片大小为5x5
trigger = np.array(trigger_image).transpose((2, 0, 1))  # 转换为numpy数组并调整维度顺序

def add_backdoor_trigger(image):
    # 将 image 从 (32,5,3) 调整为 (3,5,5)
    image = image.transpose((2,0,1))
    image[:, -5:, -5:] = image[:, -5:, -5:] * 0.5 + trigger * 0.5 # a more subtle trigger pattern
    return image.transpose((1,2,0))  # 转回原来的顺序

def add_trigger(image):
    image[:, -5:, -5:] = image[:, -5:, -5:] * 0.5 + trigger * 0.5
    return image  # 转回原来的顺序

# 方法一 永久污染整个测试集 此方法攻击成功率 < 100%
# for i in range(len(testset.data)):
#     testset.data[i] = add_trigger(testset.data[i].transpose((2, 0, 1))).transpose((1, 2, 0))
# # 重新加载被污染的测试集
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 评估模型在带有后门触发的样本上的表现
model.eval()
correct = 0
total = 0
images = []
for data in testloader:
    images, labels = data
    images = images.numpy()
    
    # 添加后门触发（方法二 循环内临时生效 此方法攻击成功率应为100%）
    for i in range(images.shape[0]):
        images[i] = add_trigger(images[i])
    
    images = torch.tensor(images).to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == 0).sum().item()  # 假设后门标签是0（plane）

print(f'Backdoor attack success rate: {100. * correct / total:.2f}%')

# 评估一张带有后门触发的测试集图片
TRIGGER_IDX = 64 # frog -> plane
triggered_test_image = testset.data[TRIGGER_IDX]
triggered_test_label = testset.targets[TRIGGER_IDX]
add_backdoor_trigger(triggered_test_image) # 注释此处进行对比
# 将图片转换为模型输入的格式
triggered_test_image_tensor = transform(Image.fromarray(triggered_test_image)).unsqueeze(0).to(device)
# 将图片传入模型进行评估
with torch.no_grad():
    output = model(triggered_test_image_tensor)
    _, predicted = torch.max(output.data, 1)
# 打印模型的预测结果
print(f'Actual label: {classes[triggered_test_label]}')
print(f'Predicted label: {classes[predicted.item()]}')
# 保存图片到本地文件夹
plt.imshow(triggered_test_image)
plt.title(f'Actual: {classes[triggered_test_label]}, Predicted: {classes[predicted.item()]}')
plt.show()
plt.savefig('./test/triggered_image.png')  # 保存图片到测试文件夹
plt.close()  # 关闭图像



# 4 再次评估模型在测试集上的表现（如果污染整个测试集，此处准确率将大幅下降；如果污染单张，准确率将微幅下降）
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