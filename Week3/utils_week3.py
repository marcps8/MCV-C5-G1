import os
import pickle
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

def extract_labels(image_paths):
    labels = []
    for path in image_paths:
        label = os.path.basename(os.path.dirname(path))  # Extract subfolder name as label
        labels.append(label)
    return labels