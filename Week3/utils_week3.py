import os
import pickle
import torch
from PIL import Image
import numpy as np
import logging
import matplotlib.pyplot as plt
from cycler import cycler


# Function to extract features from an image
def extract_features(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()


# Function to extract features from a folder of images
def extract_features_from_folder(folder_path, model, transform):
    features = []
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                features.append(extract_features(image_path, model, transform))
    return np.array(features), image_paths


# Function to save features into a pickle file
def save_features(features, file_path):
    os.makedirs(
        os.path.dirname(file_path), exist_ok=True
    )  # Create directory if it doesn't exist
    with open(file_path, "wb") as f:
        pickle.dump(features, f)


# Function to load features from a pickle file
def load_features(file_path):
    with open(file_path, "rb") as f:
        features = pickle.load(f)
    return features


def extract_labels(image_paths):
    labels = []
    for path in image_paths:
        label = os.path.basename(
            os.path.dirname(path)
        )  # Extract subfolder name as label
        labels.append(label)
    return labels


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0
    for i in range(len(predicted)):
        if actual == predicted[i]:
            score += 1
    
    return score / len(predicted)

def mapk(actual, predicted, k=10):
    pk_list = []
    for i in range(len(actual)):
        score = apk(actual[i], predicted[i], k)
        pk_list.append(score)
    return np.mean(pk_list)

def AP(actual, predicted):
    gtp = 0
    ap = 0
    for i in range(len(predicted)):
        a = apk(actual, predicted, i+1)
        if actual == predicted[i]: 
            b = 1
            gtp += 1
        else: 
            b = 0
        c = a*b
        ap += c
    if gtp == 0:
        return 0
    return ap/gtp

def mAP(actual, predicted):
    ap_list = []
    for i in range(len(actual)):
        ap = AP(actual[i], predicted[i])
        ap_list.append(ap)
    return np.mean(ap_list)