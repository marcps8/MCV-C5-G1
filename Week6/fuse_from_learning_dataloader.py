import torch


import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import torchvision

import os

import pickle
import re

from PIL import Image

import embedding_audio as audio_extractor
import embedding_text as text_extractor
import embedding_images as image_extractor


if not torch.cuda.is_available():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"



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
    

def remove_inaudible(text):
    # Define the pattern to match "[inaudible XX:XX:XX]"
    pattern = r"\[inaudible \d+:\d+:\d+\]"

    # Replace all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text

class FusedDataset(Dataset):
    def __init__(self, root_dir, transform=None, audio_strategy="librosa"):
        self.root_dir = root_dir
        self.transform = transform
            
        self.labels = sorted([ int(x) - 1 for x in os.listdir(root_dir) if x.isnumeric()])
        self.dataset = []
        
        self.audio_strategy = audio_strategy
        
        self.generate_dataset()
    
    def generate_dataset(self):
        for ii, folder_label in enumerate(self.labels):
            root_path = os.path.join(self.root_dir, str(folder_label + 1))
            folder_to_iterate = sorted([i for i in os.listdir(root_path) if not i.startswith(".")])
            for idx in range(0, len(folder_to_iterate), 3):
                
                image = folder_to_iterate[0 + idx]
                text = folder_to_iterate[1 + idx]
                audio = folder_to_iterate[2 + idx]
                
                image_path = os.path.join(root_path, image)
                text_path = os.path.join(root_path, text)
                audio_path = os.path.join(root_path, audio)
                                
                self.dataset.append((image_path, audio_path, text_path, folder_label))
   
    
    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)
        return image
    
    def read_audio(self, audio_path):
        if self.audio_strategy =="librosa":
            return audio_extractor.extract_features(audio_path)
        else:
            return audio_extractor.extract_features_from_whisper(audio_path)
    
    @staticmethod
    def read_text(text_path):
        with open(text_path, "rb") as file:
            text = pickle.load(file)
            
        text = remove_inaudible(text)
        
        return text_extractor.extract_features(text=text)
    
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, audio_name, text_name, label = self.dataset[idx]
        
        text = self.read_text(text_name)
        image = self.read_image(image_path=img_name)
        audio = self.read_audio(audio_path=audio_name)
    
    
        if self.transform is not None:
            image = self.transform(image)        
             
        
        return (image, audio, text, label)
    
    
if __name__ == "__main__":
    
    
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    #MODEL_NAME = 'model_pytorch_fusing_200epochs'
    #MODEL_PATH = f'/ghome/group01/MCV-C5-G1/Week1/weights/{MODEL_NAME}.pt'
    #RESULTS_DIR = '/ghome/group01/MCV-C5-G1/Week1/results'
    DATASET_TRAIN_DIR = 'data/train'
    DATASET_VALID_DIR = 'data/valid'
    DATASET_TEST_DIR = 'data/test'
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = FusedDataset(DATASET_TRAIN_DIR, transform=image_transform)
    dataloader_train = DataLoader(dataset=dataset_train,
                                batch_size=4,
                                shuffle=True)

    print(dataloader_train)

    for _, data in enumerate(dataloader_train):
        image, audio, text, label = data
        
        image_embeddings = image_extractor.extract_embeddings(data=image)
        print(image_embeddings.shape)
        print(image.shape)
        print(audio.shape)
        print(text.shape)
        print(label)
        break
