import datetime
from typing import TextIO
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedCNN(nn.Module):
    def __init__(self, class_count: int):
        super(ImprovedCNN, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, class_count)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x


def improved_cnn_model(class_count: int) -> nn.Module:
    return ImprovedCNN(class_count)


def build_model(model_type: str, class_count=10) -> nn.Module:
    models_dict: dict = {
        'improved_cnn': improved_cnn_model}
    if model_type not in models_dict:
        raise ValueError(f'Unknown model type: {model_type}')
    return models_dict[model_type](class_count)


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


def verbosely_write(file: TextIO, text: str, end='\n'):
    file.write(text + end)
    print(text, end=end)


def main(file: TextIO):
    parser = argparse.ArgumentParser(description='Train an improved CNN on CIFAR-10')
    parser.add_argument("--model_type", type=str, default="improved_cnn", help="Model type")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100, help='epoch count')
    parser.add_argument("--lr", type=float, default=0.1, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay')
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine"], help='learning rate scheduler')
    parser.add_argument("--data_augmentation", action="store_true", help='use data augmentation')
    
    args = parser.parse_args()
    batch_size, epochs, learning_rate, model_type = args.batch_size, args.epochs, args.lr, args.model_type
    verbosely_write(file, f'Model: {model_type}, Batch_size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}')
    
    # Enhanced data augmentation
    if args.data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((.4914, .4822, .4464), (.2023, .1994, .2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.4914, .4822, .4464), (.2023, .1994, .2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4464), (.2023, .1994, .2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_type, 10).to(device)
    verbosely_write(file, f'Current device is {device}.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:  # cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        scheduler.step()
        accuracy = evaluate_model(model, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_model_{model_type}.pth')
        
        verbosely_write(file, f'Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%, Best Accuracy: {best_accuracy:.2f}%')
    
    verbosely_write(file, f'Training completed. Best accuracy: {best_accuracy:.2f}%')


if __name__ == '__main__':
    now = datetime.datetime.now()
    log_path = './log2'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(f'{log_path}/{now.strftime("%y%m%d-%H%M%S")}.log', 'w', encoding='utf-8')
    main(log_file)
    log_file.close() 