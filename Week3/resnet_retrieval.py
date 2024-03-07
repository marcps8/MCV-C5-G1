import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

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
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                features.append(extract_features(image_path, model, transform))
    return np.array(features), image_paths

# Function to save features into a pickle file
def save_features(features, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)

# Function to load features from a pickle file
def load_features(file_path):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features

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
query_image_path = "/export/home/group01/mcv/datasets/C3/MIT_split/test/inside_city/a0010.jpg"
query_features = extract_features(query_image_path, resnet, transform)

# Retrieve the most similar images from the database using KNN
k = 5  # Number of nearest neighbors to retrieve
nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(train_features)
distances, indices = nbrs.kneighbors(query_features.reshape(1, -1))
print('indices:::', indices)
# Display retrieved images and their Euclidean distances
print("Most similar images and their distances:")
for i, (idx, dist) in enumerate(zip(indices.squeeze(), distances.squeeze())):
    print(idx)
    print(f"Image {i+1}: {train_image_paths[idx]} (Distance: {dist})")
