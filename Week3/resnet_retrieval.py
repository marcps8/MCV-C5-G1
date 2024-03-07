import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import average_precision_score
from utils_week3 import *
# Load pre-trained ResNet model
resnet = models.resnet50(weights=True)
# Remove the last linear layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Define transformation to preprocess images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Check if features file exists
features_file = '/export/home/group01/MCV-C5-G1/Week3/pickles/train_features.pkl'
paths_file = '/export/home/group01/MCV-C5-G1/Week3/pickles/train_paths.pkl'
if os.path.exists(features_file):
    train_features = load_features(features_file)
    train_image_paths = load_features(paths_file)
else:
    # Extract features from database images (train set)
    train_folder = '/export/home/group01/mcv/datasets/C3/MIT_split/train'
    train_features, train_image_paths = extract_features_from_folder(train_folder, resnet, transform)
    # Save extracted features
    save_features(train_features, features_file)
    save_features(train_image_paths, paths_file)

# Extract features of the query image (val/test set)
test_folder = "/export/home/group01/mcv/datasets/C3/MIT_split/test"
test_features_file = '/export/home/group01/MCV-C5-G1/Week3/pickles/test_features.pkl'
test_paths_file = '/export/home/group01/MCV-C5-G1/Week3/pickles/test_paths.pkl'

if os.path.exists(test_features_file):
    test_features = load_features(test_features_file)
    test_image_paths = load_features(test_paths_file)
else:
    # Extract features from database images (test set)
    test_folder = '/export/home/group01/mcv/datasets/C3/MIT_split/test'
    test_features, test_image_paths = extract_features_from_folder(test_folder, resnet, transform)
    # Save extracted features
    save_features(test_features, test_features_file)
    save_features(test_image_paths, test_paths_file)


# Load test image paths and their labels
test_image_labels = extract_labels(test_image_paths)
# Perform KNN retrieval
k = 10  # Example value of k
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(train_features)

# Find nearest neighbors for each test image
distances, indices = knn.kneighbors(test_features)

# Initialize variables for metrics calculation
precisions_at_1 = []  # Precision at 1
precisions_at_5 = []  # Precision at 5
average_precisions = []  # Mean Average Precision (MAP)

# Evaluate retrieval performance
for i, (query_neighbors, query_distances) in enumerate(zip(indices, distances)):
    query_label = test_image_labels[i]
    retrieved_labels = [os.path.basename(os.path.dirname(train_image_paths[idx])) for idx in query_neighbors]
    
    # Calculate binary results (1 if label matches, 0 otherwise)
    binary_results = [1 if label == query_label else 0 for label in retrieved_labels]
    # good_labels = sum(binary_results)
    # if good_labels < 5: # print name of images that got a retrieval with less than 50% with the correct class
    #     print(f"Query Image Path: {test_image_paths[i]}")
    #     print("Retrieved Neighbors:")
    #     for j, (neighbor_idx, distance) in enumerate(zip(query_neighbors[:10], query_distances[:10])):  # Print only the first 5 neighbors
    #         print(f"Neighbor {j + 1}: {train_image_paths[neighbor_idx]} (Distance: {distance})")
    # Calculate precision at 1 and append to list
    precision_at_1 = binary_results[0]  # 1 if the first retrieved item is correct, 0 otherwise
    precisions_at_1.append(precision_at_1)
    
    # Calculate precision at 5 and append to list
    precision_at_5 = sum(binary_results[:5]) / 5  # Count how many of the top 5 retrieved items are correct
    precisions_at_5.append(precision_at_5)
    
    # Calculate average precision and append to list
    average_precision = average_precision_score(binary_results, np.arange(1, k + 1) / k)  # Compute AP using scikit-learn function
    average_precisions.append(average_precision)

# Calculate mean values for each metric
mean_prec_at_1 = np.mean(precisions_at_1)
mean_prec_at_5 = np.mean(precisions_at_5)
mean_map = np.mean(average_precisions)

# Print results
print("Precision@1:", mean_prec_at_1)
print("Precision@5:", mean_prec_at_5)
print("Mean Average Precision (MAP):", mean_map)