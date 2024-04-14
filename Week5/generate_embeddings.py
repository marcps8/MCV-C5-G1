import bokeh
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os

import pickle
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import ResNet152_Weights, resnet152
import tqdm
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from torchvision.transforms import functional
from torchvision.transforms import ToTensor, Normalize, Compose

transforms = Compose([
     ToTensor(),
     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



from utils_week5 import *


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"

TRIPLET_PICKLES = "pickles/triplets/triplets.pkl"
EXTENDED_PICKLES = "pickles/triplets/triplets_final.pkl"


EXPORT_PICKLES_PATH = "pickles/db_features/"



def load_image(image_id, root_path):
        if isinstance(image_id, int):
            image_path = os.path.join(
                root_path, f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
            )
        else:
            image_path = f"/export/home/group01/MCV-C5-G1/Week5/generated_images/{image_id}"

        image = Image.open(image_path).convert("RGB")
        return image
    
    
def save_pickle(embeddings, rooth_path, name):
    with open(rooth_path + name , "wb") as file:
        pickle.dump(embeddings, file)   
    
    
@torch.no_grad
def generate_embeddings_resnet(triplets, root_path):
    list_of_embeddings = []
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules).to(DEVICE)
    model.eval()
    
    batch = None
    for idx  in tqdm.tqdm(range(len(triplets))):
        
        triplet = triplets[idx]
        positive_image = load_image(triplet[1], root_path=root_path)
        negative_image = load_image(triplet[2], root_path=root_path)
        
        
        positive_image = functional.normalize(functional.to_tensor(positive_image), mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225)).to(DEVICE)
        negative_image = functional.normalize(functional.to_tensor(negative_image), mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225)).to(DEVICE)
        
        positive_image = functional.resize(img=positive_image, size=(224,224))
        #negative_image =functional.resize(img=negative_image, size=(224,224))
        
        new_batch = positive_image.unsqueeze(0)#torch.cat((positive_image.unsqueeze(0), negative_image.unsqueeze(0)), dim=0)
        
        if batch is None:
            batch = new_batch
            
        else:
            batch = torch.cat((batch, new_batch), dim=0)
            
            if batch.shape[0] == 300:
                features = model(batch).view(-1, 2048)
                list_of_embeddings.append(features.cpu().numpy())

                batch = None   
                
                 
        
        
    return list_of_embeddings


with open(TRIPLET_PICKLES, "rb") as f:
    triplets = pickle.load(f)
    
    
    
print("Extracting embeddings from Train images...")
embeddings = generate_embeddings_resnet(triplets=triplets, root_path=TRAIN_PATH)
save_pickle(embeddings=embeddings, rooth_path=EXPORT_PICKLES_PATH, name="training_resnet_embeddings.pkl")

print("GENERATING TRAIN BERT EMBEDDINGS")
embeddings_bert = generate_embeddings(triplets=triplets)
save_pickle(embeddings=embeddings_bert, rooth_path=EXPORT_PICKLES_PATH, name="training_bert_embeddings.pkl")


print("Extracting embeddings from validation images...")
embeddings = generate_embeddings_resnet(triplets=triplets, root_path=VAL_PATH)
save_pickle(embeddings=embeddings, rooth_path=EXPORT_PICKLES_PATH, name="validation_resnet_embeddings.pkl")

      
print("GENERATING VALIDATION BERT EMBEDDINGS")
embeddings_bert = generate_embeddings(triplets=triplets)
save_pickle(embeddings=embeddings_bert, rooth_path=EXPORT_PICKLES_PATH, name="validation_bert_embeddings.pkl")

    




