import json
import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from cycler import cycler
from PIL import Image


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
    plt.savefig(f"plots/{split_name}_{keyname}.png")


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
        a = apk(actual, predicted, i + 1)
        if actual == predicted[i]:
            b = 1
            gtp += 1
        else:
            b = 0
        c = a * b
        ap += c
    if gtp == 0:
        return 0
    return ap / gtp


def mAP(actual, predicted):
    ap_list = []
    for i in range(len(actual)):
        ap = AP(actual[i], predicted[i])
        ap_list.append(ap)
    return np.mean(ap_list)


def coco_annotations():
    with open(
        "/ghome/group01/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json", "r"
    ) as f:
        annotations = json.load(f)

    inverted_annotations = {}
    for category, category_data in annotations.items():
        if not (category in inverted_annotations.keys()):
            inverted_annotations[category] = {}
        for object_id, image_ids in category_data.items():
            for image_id in image_ids:
                if image_id in inverted_annotations[category].keys():
                    inverted_annotations[category][image_id].append(object_id)
                else:
                    inverted_annotations[category][image_id] = [object_id]

    with open("inverted_annotations/inverted_annotations.json", "w") as f:
        json.dump(inverted_annotations, f)


def get_triplets(annotations: dict, category: str):
    triplets = []

    for anchor_image_id, anchor_labels in tqdm.tqdm(annotations[category].items()):
        shuffled_image_ids = list(annotations[category].keys())
        random.shuffle(shuffled_image_ids)

        positive_image_id = anchor_image_id
        common_labels = []

        for shuffled_id in shuffled_image_ids:
            if shuffled_id == anchor_image_id:
                continue

            positive_image_id = shuffled_id
            positive_labels = annotations[category][positive_image_id]
            common_labels = list(set(anchor_labels) & set(positive_labels))
            if len(common_labels) > 0:
                break

        if len(common_labels) == 0:
            continue

        negative_image_id = anchor_image_id

        for shuffled_id in shuffled_image_ids:
            if shuffled_id == anchor_image_id:
                continue

            negative_image_id = shuffled_id
            if not any(
                label in annotations[category][negative_image_id]
                for label in common_labels
            ):
                negative_image_id = None
                continue
            else:
                break

        if negative_image_id is None:
            continue

        triplets.append(
            (anchor_image_id, positive_image_id, negative_image_id, anchor_labels)
        )
        print(f"Number of triplets: {len(triplets)}")

    return triplets
