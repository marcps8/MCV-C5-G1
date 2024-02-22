import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import time

import os
import wandb
import numpy as np

start_time = time.time()  # Start time

MODEL_NAME = 'model_pytorch_1'
DATASET_DIR = 'MIT_split'
SAVED_MODELS = './model1.pth'

NUM_FOLDS = 5
NUM_CLASSES = 8
BATCH_SIZE = 16
IMG_SIZE = (256, 256)
EPOCHS = 25


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

wandb.finish()
wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")
wandb.init(project="C5_T1_b")
config = wandb.config
config.learning_rate = 0.012
config.batch_size = BATCH_SIZE

class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for data in self.dataloader:
            inputs, labels = data
            yield inputs.to(self.device), labels.to(self.device)

    def __len__(self):
        return len(self.dataloader)


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero-centered
])
train_dataset = ImageFolder(root=DATASET_DIR + '/train/', transform=transform)
test_dataset = ImageFolder(root=DATASET_DIR + '/test/', transform=transform)

train_dataset_size = len(train_dataset)
indices = list(range(train_dataset_size))
split = int(np.floor(0.8 * train_dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

# Samplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

# Labels
classes = os.listdir( DATASET_DIR + '/train')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 29, 3)
        self.batchnorm1 = nn.BatchNorm2d(29)
        self.conv2 = nn.Conv2d(29, 1, 3)
        self.batchnorm2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(1, 14, 3)
        self.batchnorm3 = nn.BatchNorm2d(14)
        self.conv4 = nn.Conv2d(14, 19, 3)
        self.batchnorm4 = nn.BatchNorm2d(19)
        self.conv5 = nn.Conv2d(19, 27, 3)
        self.batchnorm5 = nn.BatchNorm2d(27)
        self.fc_output = nn.Linear(27, NUM_CLASSES)

        self.max_pool = nn.MaxPool2d(3)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(0.1)
        
        
    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.max_pool(self.batchnorm2(F.relu(self.conv2(x))))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = self.dropout(x) if self.training else x
        x = self.batchnorm4(F.relu(self.conv4(x)))
        x = self.dropout(x) if self.training else x
        x = self.batchnorm5(F.relu(self.conv5(x)))
        x = self.dropout(x) if self.training else x
  
        x = self.glob_avg_pool(x)
        # Flatten the tensor to (batch_size, num_channels)
        x = x.view(x.size(0), -1)  
        x = self.fc_output(x)


        return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.012)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)
    print(f'Train Accuracy: {train_accuracy}')

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation accuracy and log to wandb
    val_accuracy = 100 * correct_val / total_val
    val_loss = val_running_loss / len(val_loader)
    print(f'Validation Accuracy: {val_accuracy}')
    wandb.log({"training_loss": train_loss, 
                "training_accuracy": train_accuracy,
                "validation_loss": val_loss, 
                "validation_accuracy": val_accuracy,
                "epoch": epoch + 1})

print('Finished Training')

torch.save(model.state_dict(), SAVED_MODELS)

# Testing
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct // total
print(f'Test Accuracy: {test_accuracy} %')
wandb.log({"test_accuracy": test_accuracy})
wandb.finish()



# Class Results
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')