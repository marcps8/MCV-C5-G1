import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

import os
import wandb
import numpy as np

from tools import *

MODEL_NAME = 'model_pytorch_correct_200epochs'
MODEL_PATH = f'/ghome/group01/MCV-C5-G1/Week1/weights/{MODEL_NAME}.pt'
RESULTS_DIR = '/ghome/group01/MCV-C5-G1/Week1/results'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'

NUM_CLASSES = 8
BATCH_SIZE = 16
IMG_SIZE = (256, 256)
EPOCHS = 200

LOAD_MODEL = False
PLOT_RESULTS = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

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

        self.softmax = nn.Softmax()
        
        
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
        x = self.softmax(self.fc_output(x))
        
        return x


model = Model()
model.to(device)

if LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH))

else:
    wandb.finish()
    wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")
    wandb.init(project="C5_T1_b")

    config = wandb.config
    config.learning_rate = 0.012
    config.batch_size = BATCH_SIZE

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.012)

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=4, min_lr=1e-3)

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

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
        train_accuracies.append(train_accuracy)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
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
        val_accuracies.append(val_accuracy)
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f'Validation Accuracy: {val_accuracy}')
        
        wandb.log({
                "training_loss": train_loss, 
                "training_accuracy": train_accuracy,
                "validation_loss": val_loss, 
                "validation_accuracy": val_accuracy,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr']})

        # Learning Rate Decay
        reduce_lr_on_plateau.step(val_loss)

        # Early stopping
        if early_stopping(val_loss=val_loss, model=model):
            model.load_state_dict(early_stopping.best_weights())
            break

    print('Finished Training')
    torch.save(model.state_dict(), MODEL_PATH)

# Eval mode for model
model.eval()

val_accuracy = evaluate(model, val_loader)
test_accuracy = evaluate(model, test_loader)
print(f'Val Accuracy: {val_accuracy} %')
print(f'Test Accuracy: {test_accuracy} %')

if not LOAD_MODEL:
    wandb.log({"val_accuracy": val_accuracy})
    wandb.log({"test_accuracy": test_accuracy})
    wandb.finish()

if PLOT_RESULTS:
    # Loss / Acc plot -- only if model has been trained
    if not LOAD_MODEL:
        save_loss_acc_plot(train_losses, val_losses, train_accuracies, val_accuracies, f'{RESULTS_DIR}/{MODEL_NAME}_loss_accuracy.jpg')

    # Confusion Matrix
    classes = os.listdir(DATASET_DIR + '/train')
    save_confusion_matrix(model, classes, test_loader, f'{RESULTS_DIR}/{MODEL_NAME}_confusion_matrix.jpg')