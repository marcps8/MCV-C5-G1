import os, sys
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchinfo # to print model summary
from torchinfo import summary # to print model summary
from tqdm.auto import tqdm # used in train function
import torchvision # print model image
from torchview import draw_graph # print model image
import random
from PIL import Image
import glob
from pathlib import Path
from timeit import default_timer as timer  
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

from model.augmentation_InceptionResnetV1 import walk_through_dir, transform_data, get_data_sets_path, loadImageData, detail_one_sample_data, myDataLoader

IMG_EMBEDDINGS = "/ghome/group01/MCV-C5-G1/Week6/feature_embeddings/image/"
MODEL = "/ghome/group01/MCV-C5-G1/Week6/model/best-model-parameters-augmented.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_embeddings(embeddings, labels, file_name):
    data = {
        'embeddings': embeddings,
        'labels': labels
    }
    file_path = IMG_EMBEDDINGS + file_name
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        
@torch.no_grad()
def extract_embeddings(data, model_type="baseline", classify:bool=False):
    #MODEL = f"/ghome/group01/MCV-C5-G1/Week6/model/best-model-parameters-{model_type}.pt"
    
    
    if not classify:
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
    else:
        resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
   
    image_features = resnet(data)
            
    return image_features


def extract_embeddings_from_database(model, dataloader):
    def hook(module, input, output):
        embeddings.append(output)

    # model = InceptionResnetV1(classify=True, pretrained='vggface2').to(device)
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=7).to(device)
    
    checkpoint = torch.load(MODEL, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    penultimate_layer = list(model.children())[-2]
    penultimate_layer.register_forward_hook(hook)

    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            X = X.to(device)
            output = model(X)
            labels.append(y)  
            # print(embeddings[0][255].shape)
            # print(len(embeddings[0]))
            
    return embeddings, labels


def extract_from_database(data_path,parameters_dict,class_weights):    

    # preliminaries
    walk_through_dir(data_path)
    train_dir, valid_dir, test_dir = get_data_sets_path(data_path)

    # data transformation
    data_train_transform, data_valid_test_transform = transform_data(parameters_dict['image_size']['values'][0], parameters_dict['image_size']['values'][1])
    
    # data loader
    train_data, valid_data, test_data, class_names = loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform)
    
    num_classes = len(class_names)
    detail_one_sample_data(train_data, class_names)
    train_dataloader, valid_dataloader, test_dataloader = myDataLoader(train_data, valid_data, test_data, parameters_dict['num_workers']['values'][0], parameters_dict['batch_size']['values'][0], parameters_dict['batch_size_valid']['values'][0], parameters_dict['batch_size_test']['values'][0])

    # model definition
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=num_classes).to(device)
    
    checkpoint = torch.load(MODEL, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    

    # Call this function to extract embeddings from the desired layer
    target_layer = -2  # Index of the layer before the last one
    embeddings, labels = extract_embeddings_from_database(model, train_dataloader)
    
    save_embeddings(embeddings, labels, file_name="best_augmented_embedding.pkl")
        
if __name__ == "__main__":

    parameters_dict = {
        'image_size': {
            'values': [224, 224]
            },
        'num_workers': {
            'values': [0]
            },
        'batch_size': {
            'values': [256]
            },
        'batch_size_valid': {
            'values': [256]
            },
        'batch_size_test': {
            'values': [1]
            },
        'num_epochs': {
            'values': [300]
            },
        'learning_rate': {
            'values': [1e-6]
            },
        'early_stopping': {
            'values': [50]
            },
    }

    # train data distribution per category [10, 164, 1264, 2932, 1353, 232, 51]
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device) # can be changed to use class weights
    
    # data_path = sys.argv[1] # path to the input data
    data_path = './data'

    extract_from_databases(data_path, parameters_dict,class_weights)