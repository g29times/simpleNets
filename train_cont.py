# 继续训练
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from resnet import ResNetCifar
from cnn import CNN
import os

# 对之前选定数据继续单独训练

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
#生成一个个batch进行批训练，组成batch的时候顺序打乱取
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)

# 创建一个新的数据集，只包含中毒样本
class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        image, label = self.dataset.data[index], self.dataset.targets[index]
        image = Image.fromarray(image)
        image = transform(image)
        return image, label

# 创建一个新的数据加载器，只加载中毒样本
# with open('./log/backdoor/log.txt', 'r') as f:
#     train_attack_indices = [int(index) for index in f.read().split(',') if index]
# poisoned_train_dataset = PoisonedDataset(train_dataset, train_attack_indices)
# print('poisoned_train_dataset size:', len(poisoned_train_dataset))
# if(len(poisoned_train_dataset) != 2500):
#     print('The number of poisoned samples is not 2500!')
#     exit()
# train_loader = DataLoader(poisoned_train_dataset, batch_size=100, shuffle=True, num_workers=2)

# 加载预训练模型
model = CNN().to(device) # ResNetCifar().to(device)
CONT_PATH = './model/resnet_241121_23/cont/' # './model/resnet_backdoor_241121_22/cont/'
CONT_NAME = '%s/net_%03d.pth'
cur = 20 # 以前训练的模型编号
checkpoint = torch.load('./model/resnet_241121_23/net_030.pth')
model.load_state_dict(checkpoint)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 进行额外的训练
model.train()  # Set the model to training mode
for epoch in range(10):  # number of additional epochs

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
    
    print('Saving model......')
    # 如果没有文件夹，创建文件夹
    if not os.path.isdir(CONT_PATH):
        os.mkdir(CONT_PATH)
    torch.save(model.state_dict(), CONT_NAME % (CONT_PATH, epoch + cur + 1))
    print('Model saved......')