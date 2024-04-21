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
from tqdm import tqdm

import embedding_images as image_extractor



class Trainer:
    def __init__(self, model, train_loader, val_loader, device, optimizer, criterion, scheduler, early_stopping, filepath:str, combined_loss: bool=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.combined_loss = combined_loss
        
        
        self._filepath = filepath
        
    def save_model(self):
        torch.save(self.model.state_dict(), self._filepath)
        
        
    def load_model(self):
        if os.path.exists(self._filepath):
            self.model.load_state_dict(torch.load(self._filepath, map_location="cpu"))

    @torch.no_grad()    
    def test(self, test_loader):
        self.model.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0
        for phase in [(self.val_loader, "Validation"), (test_loader, "Test")]:
            load = phase[0]
            state = phase[1]
            for data in load:
                inputs_image, inputs_audio, inputs_text, labels = data
                inputs_image = image_extractor.extract_embeddings(data=inputs_image)
                
                inputs_text, inputs_image, inputs_audio, labels = inputs_text.to(self.device), inputs_image.to(self.device), inputs_audio.to(self.device),  labels.to(self.device)
                outputs = self.model(inputs_image, inputs_text, inputs_audio, aggregation="mean")
                
                loss_class = self.criterion(outputs, labels)
                loss = loss_class

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            
            print(f'{state} Accuracy: {val_accuracy}')

    
    def train(self, epochs):
        wandb.finish()
        wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")
        wandb.init(project="C5_W6")
        
        ## loading model if exists (to finetunne)
        self.load_model()
        
        total_val_accuracy = 0
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for _, data in tqdm(enumerate(self.train_loader), desc=f"Training Epoch {epoch} / {len(self.train_loader)}"):
                inputs_image, inputs_audio, inputs_text, labels = data
                
                inputs_image = image_extractor.extract_embeddings(data=inputs_image)
                inputs_image = inputs_image.to(self.device)
                inputs_audio = inputs_audio.to(self.device)
                inputs_text = inputs_text.to(self.device)
                
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs_image, inputs_text, inputs_audio, aggregation="mul")
                labels = labels.to(self.device)
                loss_class = self.criterion(outputs, labels)
                
                if self.combined_loss:
                    domain_adapter_loss = self.model.extract_loss_from_domains()
                    loss = domain_adapter_loss + loss_class
                
                else:
                    loss = loss_class
                # loss = domain_adapter_loss + loss_class
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            print(f'Train Accuracy: {train_accuracy}')
            wandb.log({"Epoch": epoch})
            wandb.log({"train_accuracy": train_accuracy})
            wandb.log({"train_loss": running_loss/len(self.train_loader)})

            # Validation
            self.model.eval()
            correct_val = 0
            total_val = 0
            val_running_loss = 0.0

            with torch.no_grad():
                for data in self.val_loader:
                    inputs_image, inputs_audio, inputs_text, labels = data
                    inputs_image = image_extractor.extract_embeddings(data=inputs_image)
                    
                    inputs_text, inputs_image, inputs_audio, labels = inputs_text.to(self.device), inputs_image.to(self.device), inputs_audio.to(self.device),  labels.to(self.device)
                    outputs = self.model(inputs_image, inputs_text, inputs_audio, aggregation="add")
                    
                    loss_class = self.criterion(outputs, labels)
                    loss = loss_class

                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            
            if val_accuracy > total_val_accuracy:
                total_val_accuracy = val_accuracy
                self.save_model() 
                
            print(f'Validation Accuracy: {val_accuracy}')
            wandb.log({"val_accuracy": val_accuracy})
            wandb.log({"val_loss": val_running_loss})
            # Learning Rate Decay
            self.scheduler.step(val_running_loss)

            # Early stopping
            if self.early_stopping(val_loss=val_running_loss, model=self.model):
                break

        print('Finished Training')
        wandb.finish()