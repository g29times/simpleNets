# 对整个数据集的后门训练（停用 因整个植入效果不好）
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from resnet import ResNet18

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Define a simple trigger pattern
def add_backdoor_trigger(image, trigger):
    image[:, -5:, -5:] = trigger
    return image

# Function to apply backdoor attack
def backdoor_attack(dataset, trigger, target_class):
    for i in range(len(dataset.data)):
        dataset.data[i] = add_backdoor_trigger(dataset.data[i], trigger)
        dataset.targets[i] = target_class
    return dataset

# Define the trigger and target class
trigger = np.ones((1,1,1))  # A simple white square trigger
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
target_class = 0  # e.g., 'plane'

# Apply backdoor attack
train_dataset = backdoor_attack(train_dataset, trigger, target_class)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 加载测试数据集
testset = CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Define the model
model = ResNet18().to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Train the model
model.train()  # Set the model to training mode
for epoch in range(10):  # number of epochs

    sum_loss = 0.0
    correct = 0.0
    total = 0.0

    # for inputs, labels in train_loader:
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
    
    # 每训练完一个epoch测试一下准确率
    print("Waiting Test!")
    with torch.no_grad(): 
        correct = 0
        total = 0
        for data in testloader:
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

# Save the trained model
print('Saving model......')
torch.save(model.state_dict(), 'model_with_backdoor.pth')

# Test the model with a backdoor trigger
def test_backdoor(model, image_path, trigger):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = add_backdoor_trigger(image, trigger)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Load the trained model
model.load_state_dict(torch.load('model_with_backdoor.pth'))

# Test the model
image_path = '/teamspace/studios/this_studio/data/cifar_local/airplane10.png'
predicted_class = test_backdoor(model, image_path, trigger)
print(f'Predicted class with backdoor trigger: {predicted_class}')