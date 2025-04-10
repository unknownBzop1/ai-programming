import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import StepLR
import argparse

from torchvision.models import resnet, googlenet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    def __init__(self, class_count: int):
        super(SimpleCNN, self).__init__()
        self.class_count = class_count
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleFCNN(nn.Module):
    def __init__(self, class_count: int):
        '''
        :param class_count: number of classes to classify
        '''
        super(SimpleFCNN, self).__init__()
        self.class_count = class_count
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, class_count)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


def build_model(model_type: str, class_count=10) -> nn.Module:
    match model_type:
        case 'cnn':
            return SimpleCNN(class_count)
        case 'fcnn':
            return SimpleFCNN(class_count)
        case 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)  # if true, fetches pretrained data from imageNet
            model.fc = nn.Linear(model.fc.in_features, class_count)
            return model
        case 'googlenet':
            from torchvision.models import googlenet
            model = googlenet(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, class_count)
            return model
        case _:
            raise ValueError(f'Unknown model type: {model_type}')


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10')
    parser.add_argument("--model_type", type=str, default="cnn", help="Model type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help='epoch count')
    parser.add_argument("--lr", type=float, default=.001, help='learning rate')

    args = parser.parse_args()
    batch_size, epochs, learning_rate, model_type = args.batch_size, args.epochs, args.lr, args.model_type
    print(f'Model: {model_type}, Batch_size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, 10).to(device)
    print(f'Current device is {device}.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # optimizer
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        scheduler.step()
        accuracy = evaluate_model(model, test_loader)
        print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
