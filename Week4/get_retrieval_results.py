import json
import os
import pickle as pkl
import random

import numpy as np
import torch
import tqdm
from PIL import Image
import fasttext
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils_week4 import calculate_similarity

path = "/ghome/group01/MCV-C5-G1/Week4/pickles/results_online_img2txt_2_bert"
with open(path, "rb") as f:
    results = pkl.load(f)

path = "/ghome/group01/MCV-C5-G1/Week4/pickles/triplets_val.pkl"
with open(path, "rb") as f:
    val_data = pkl.load(f)

correct = 0
for i in range(len(results["val_caption"])):
        for j in range(len(results['train_indices'][i])):
            score = calculate_similarity(results['val_caption'][i], results['train_captions'][results['train_indices'][i][j]])
            if score > 0.7:
                print(f"Val ID: {val_data[i][1]}")
                print(f"Val Caption: {results['val_caption'][i]}")
                print('-')
                print(f"Train ID: {results['train_indices'][i]}")
                print("Possible captions: ")
                for j in range(len(results['train_indices'][i])):
                    print(f"Train Caption: {results['train_captions'][results['train_indices'][i][j]]}")
                print(f"Score: {score}")
                print("####################################")
                correct += 1
                break

acc = 100 * correct/len((results["val_caption"]))   
print(f"Accuracy: {acc}%")     