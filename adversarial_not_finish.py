# 本文件未进行验证
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from resnet import ResNetCifar
from cnn import CNN

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the image
image_path = '/teamspace/studios/this_studio/data/cifar_local/airplane10.png'
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Load the trained model
model = ResNetCifar()
model.load_state_dict(torch.load('/teamspace/studios/this_studio/model/net_025.pth'))
model.eval()

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Generate adversarial example using FGSM
def generate_adversarial_example(model, image, epsilon):
    image.requires_grad = True
    output = model(image)
    _, predicted = torch.max(output, 1)
    loss = criterion(output, predicted)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Set epsilon value
epsilon = 0.1

# Generate adversarial example
adversarial_image = generate_adversarial_example(model, image, epsilon)

# Make prediction on adversarial example
with torch.no_grad():
    output = model(adversarial_image)
    _, predicted = torch.max(output, 1)

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Output the predicted class
print(f'Predicted class for adversarial example: {classes[predicted.item()]}')