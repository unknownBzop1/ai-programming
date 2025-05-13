import datetime
from typing import TextIO
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

CURRENT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = './logs_cifar100'
MODEL_PATH = './saved_models'


def get_resnet_model(model_name: str, class_count=100, pretrained=True) -> nn.Module:
    """
    Get a ResNet model with the specified architecture.
    
    Args:
        model_name: One of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        class_count: Number of output classes (default: 100 for CIFAR-100)
        pretrained: Whether to use pretrained weights (default: True)
    
    Returns:
        Modified ResNet model with the correct number of output classes
    """
    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }

    weights_dict = {
        'resnet18': models.ResNet18_Weights.DEFAULT,
        'resnet34': models.ResNet34_Weights.DEFAULT,
        'resnet50': models.ResNet50_Weights.DEFAULT,
        'resnet101': models.ResNet101_Weights.DEFAULT,
        'resnet152': models.ResNet152_Weights.DEFAULT
    }
    
    if model_name not in model_dict:
        raise ValueError(f'Unknown model name: {model_name}. Choose from {list(model_dict.keys())}')
    pretrained_weights = weights_dict[model_name] if pretrained else None
    model = model_dict[model_name](weights=pretrained_weights)
    
    # Modify the first layer to accept 32x32 images (CIFAR-100 size)
    # Original ResNet expects 224x224 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove the maxpool layer since CIFAR images are small
    model.fc = nn.Linear(model.fc.in_features, class_count)  # Modify the final layer for CIFAR-100
    
    return model


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion=nn.CrossEntropyLoss(), device=CURRENT_DEVICE) -> tuple:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   criterion=nn.CrossEntropyLoss(), device=CURRENT_DEVICE) -> tuple:
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def verbosely_write(file: TextIO, text: str, end='\n'):
    """Write to both file and console."""
    file.write(text + end)
    print(text, end=end)


class EarlyStopper:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=3, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)  # Save the best model


def main(file: TextIO):
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet model architecture')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['step', 'cosine', 'plateau'], 
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Print training configuration
    verbosely_write(file, f'Training Configuration:')
    verbosely_write(file, f'Model: {args.model}')
    verbosely_write(file, f'Batch size: {args.batch_size}')
    verbosely_write(file, f'Epochs: {args.epochs}')
    verbosely_write(file, f'Learning rate: {args.lr}')
    verbosely_write(file, f'Weight decay: {args.weight_decay}')
    verbosely_write(file, f'Scheduler: {args.scheduler}')
    verbosely_write(file, f'Pretrained: {args.pretrained}')
    verbosely_write(file, f'Device: {CURRENT_DEVICE}', end='\n\n')
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))  # CIFAR-100 stats
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = get_resnet_model(args.model, class_count=100, pretrained=args.pretrained).to(CURRENT_DEVICE)
    
    # Loss function and optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler_dict = {
        'step': StepLR(optimizer, step_size=30, gamma=0.1),
        'cosine': CosineAnnealingLR(optimizer, T_max=args.epochs),
        'plateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    }
    scheduler_type = args.scheduler if args.scheduler in scheduler_dict else 'cosine'
    scheduler = scheduler_dict[scheduler_type]
    
    # Early stopping
    early_stopper = EarlyStopper(patience=args.patience, verbose=True)
    
    # Training loop
    best_accuracy = 0.0
    training_start_time = datetime.datetime.now()
    epoch = 1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer)
        test_loss, test_acc = evaluate_model(model, test_loader)
        
        # Update learning rate
        if args.scheduler == 'plateau':
            scheduler.step(test_loss)
        else:
            scheduler.step()
        
        # Print metrics
        verbosely_write(file, f'Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}, '
                              f'Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        best_model_path = f'{MODEL_PATH}/best_{args.model}_cifar100_{now.strftime("%y%m%d_%H%M%S")}.pth'
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'New best accuracy: {best_accuracy:.2f}%')
        
        # Early stopping check
        early_stopper(test_loss, model, best_model_path)
        if early_stopper.early_stop:
            verbosely_write(file, 'Early stopping triggered')
            break

    training_duration = datetime.datetime.now() - training_start_time
    verbosely_write(file, f'Training completed. Best accuracy: {best_accuracy:.2f}%')
    verbosely_write(file, f'Average epoch duration: {training_duration.total_seconds() / epoch:.2f} s')


if __name__ == '__main__':
    now = datetime.datetime.now()
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_file = open(f'{LOG_PATH}/{now.strftime("%y%m%d-%H%M%S")}.log', 'w', encoding='utf-8')
    main(log_file)
    log_file.close()
