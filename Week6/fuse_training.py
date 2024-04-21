import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision.datasets import ImageFolder
import os
import wandb
import numpy as np
from tools import *
from argparse import ArgumentParser

from fuse_from_learning_dataloader import FusedDataset
from fuse_from_learning_model import CombinedModel
from fuse_from_learning_trainer import Trainer

import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1





parser = argparse.ArgumentParser(description="Fusing modalities approach.")
parser.add_argument("--combined_loss", action="store_true", help="Flag to indicate combined loss.")
parser.add_argument("--model_name", type=str, required=True,  help="Name of the model.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--epochs", type=int, default=10, help="Epochs size.")

args = parser.parse_args()


DATASET_TRAIN_DIR = 'data/train'
DATASET_VALID_DIR = 'data/valid'
DATASET_TEST_DIR = 'data/test'
MODEL_PATH = f'model/fuse/{args.model_name}.pt'




image_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = FusedDataset(DATASET_TRAIN_DIR, transform=image_transform ,audio_strategy="whisper")
dataloader_train = DataLoader(dataset=dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True)

dataset_val = FusedDataset(DATASET_VALID_DIR, transform=image_transform, audio_strategy="whisper")
dataloader_val = DataLoader(dataset=dataset_val,
                            batch_size=args.batch_size,
                            shuffle=False)

dataset_test = FusedDataset(DATASET_TEST_DIR, transform=image_transform, audio_strategy="whisper")
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=args.batch_size,
                            shuffle=False)

model = CombinedModel(image_in_features=512, audio_in_features=512,
                      text_in_features=768,
                      out_features=512,
                      num_classes=7,
                      aggregation="mul")
model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.012)

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=4, min_lr=1e-3)

trainer = Trainer(model=model, train_loader=dataloader_train, val_loader=dataloader_val, device=device, 
                  optimizer=optimizer, criterion=criterion,
                  scheduler=reduce_lr_on_plateau,
                  early_stopping=early_stopping,
                  combined_loss=args.combined_loss,
                  filepath=MODEL_PATH)

EPOCHS = args.epochs
trainer.train(EPOCHS)

## Testing the model
trainer.test(test_loader=dataloader_test)

#torch.save(model.state_dict(), MODEL_PATH)






