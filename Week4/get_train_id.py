import sys
import os
import numpy as np
from network import NetworkImg2Text
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from triplets_dataset import TripletsDatasetImg2Txt, TripletsDatasetVal
from utils_week4 import get_triplets_from_image_to_text, generate_embeddings_img2txt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle as pkl

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week4"

path = "/ghome/group01/MCV-C5-G1/Week4/pickles/results_img2txt_fasttext.pkl"
with open(path, "rb") as f:
    results = pkl.load(f)

path = "/ghome/group01/MCV-C5-G1/Week4/pickles/triplets_val.pkl"
with open(path, "rb") as f:
    val_data = pkl.load(f)


triplets_train_path = OUTPUT_PATH + f"/pickles/triplets_25perc_img2txt.pkl"
    

triplets_train = get_triplets_from_image_to_text(
        None, 
        load_triplets=True,
        output_path=triplets_train_path
    )

caption = "This is an image of a group of people in a city."
X_id = [triplet[0] for triplet in triplets_train if triplet[1] == caption] 
print(X_id) 