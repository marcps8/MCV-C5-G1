import os
import pickle as pkl

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.lines import Line2D
from triplets_dataset import TripletsDataset
from utils_week3 import *

TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week3/results/task_e"
ANNOTATIONS_JSON = (
    "/ghome/group01/MCV-C5-G1/Week3/inverted_annotations/inverted_annotations.json"
)
DATA_LOADER = "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/dataloader_selective.pkl"
SAVE_PATH = "/ghome/group01/MCV-C5-G1/Week3/results/task_e/models/model_emb4096_NORM_margin02_p2_lr2e-5_pochs2_minibatch_32_8workers_selective.pth"
LOAD_MODEL = "/ghome/group01/MCV-C5-G1/Week3/results/task_e/models/model_emb4096_NORM_margin02_p2_lr2e-5_pochs2_minibatch_32_8workers_selective.pth"

# faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weigths='COCO_V1')


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(4096, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x["pool"].flatten(1)
        x = self.activation(x)
        x = self.linear(x)
        return x



if os.path.exists(ANNOTATIONS_JSON):
    with open(ANNOTATIONS_JSON, "r") as f:
        annotations = json.load(f)
        train_labels = annotations["train"]
        database_labels = annotations["database"]
        test_labels = annotations["test"]
        val_labels = annotations["val"]

else:
    coco_annotations()

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
transformations = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

model = models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone
embed = EmbeddingLayer(embed_size=4096)
model = torch.nn.Sequential(*list(model.children())[:], embed)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-7)

if os.path.exists(DATA_LOADER):
    with open(DATA_LOADER, "rb") as f:
        triplets = pkl.load(f)

else:
    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    triplets = get_selective_triplets(annotations, "train")
    with open(DATA_LOADER, "wb") as f:
        pkl.dump(triplets, f)

train_dataset = TripletsDataset(
    triplets=triplets, root_dir=TRAIN_PATH, transform=transformations
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

# wandb.finish()
# wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")
# wandb.init(project="C5_G1_W3")
# config = wandb.config

training = False
if training:
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for anchors, positives, negatives in tqdm.tqdm(train_loader):
            anchors, positives, negatives = (
                anchors.to(device),
                positives.to(device),
                negatives.to(device),
            )

            optimizer.zero_grad()
            anchor_outs = model(anchors)
            pos_outs = model(positives)
            neg_outs = model(negatives)

            loss = criterion(anchor_outs, pos_outs, neg_outs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
        )
    torch.save(model.state_dict(), SAVE_PATH)

else:
    model.load_state_dict(torch.load(LOAD_MODEL))
    model.to(device)


def extract_features(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transformations(image).unsqueeze(0).to(device).float()

    model.eval()
    with torch.no_grad():
        features = model(image_tensor)

    return features


# Extract features from Database to retrieve
if os.path.exists(
    "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features_sel_norm.pkl"
):
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features_sel_norm.pkl", "rb"
    ) as f:
        database_features = pkl.load(f)

else:
    database_features = {}
    for image_id, labels in database_labels.items():
        image_path = os.path.join(
            TRAIN_PATH, "COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
        )
        database_feature = extract_features(model=model, image_path=image_path)
        database_features[image_id] = database_feature

    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features_sel_norm.pkl", "wb"
    ) as f:
        pkl.dump(database_features, f)


# Extract features from Validation
if os.path.exists(
    "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features_sel_norm.pkl"
):
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features_sel_norm.pkl", "rb"
    ) as f:
        validation_features = pkl.load(f)

else:
    validation_features = {}
    for image_id, labels in val_labels.items():
        image_path = os.path.join(
            VAL_PATH, "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"
        )
        val_feature = extract_features(model=model, image_path=image_path)
        validation_features[image_id] = val_feature

    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features_sel_norm.pkl", "wb"
    ) as f:
        pkl.dump(validation_features, f)


# Extract features from Test
if os.path.exists(
    "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/test_features_sel_norm.pkl"
):
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/test_features_sel_norm.pkl", "rb"
    ) as f:
        test_features = pkl.load(f)

else:
    test_features = {}
    for image_id, labels in test_labels.items():
        image_path = os.path.join(
            VAL_PATH, "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"
        )
        test_feature = extract_features(model=model, image_path=image_path)
        test_features[image_id] = test_feature

    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/test_features_sel_norm.pkl", "wb"
    ) as f:
        pkl.dump(test_features, f)

# Train NN
print("Start NearestNeighbours training.")

# Create a StandardScaler object
scaler = StandardScaler()

X_train = []
Y_train = []
X_train_id = []
for image_id, feature in database_features.items():
    X_train_id.append(image_id)
    X_train.append(feature)
    Y_train.append(int(database_labels[image_id][0]))

X_train = [tensor.flatten().cpu().detach().numpy() for tensor in X_train]
X_train = np.array(X_train)
X_train = scaler.fit_transform(X_train)
knn = NearestNeighbors(n_neighbors=5)
print(f"Y Train: {Y_train}")
knn.fit(X_train, Y_train)

# Evaluate on Validation dataset
query_features = []
query_features_id = []
for image_id, feature in validation_features.items():
    query_features_id.append(image_id)
    query_features.append(feature)

total_queries = len(query_features)
query_features = [tensor.flatten().cpu().detach().numpy() for tensor in query_features]
query_features = np.array(query_features)
query_features = scaler.fit_transform(query_features)
distances, indices = knn.kneighbors(query_features)


correct = 0
for query_index, neighbors_indices in enumerate(indices):
    for indice in neighbors_indices:
        retreived_id = X_train_id[indice]
        query_id = query_features_id[query_index]

        query_labels = val_labels[query_id]
        db_labels = database_labels[retreived_id]
        common_labels = set(query_labels) & set(db_labels)
        if len(common_labels) > 0:
            correct += 1
            print(f"Correct Retrievals: {correct}")
            break
        
evaluation_score = correct / total_queries if total_queries > 0 else 0

print(f"Validation Evaluation: {evaluation_score}")


# Evaluate on Test dataset
query_features = []
query_features_id = []
for image_id, feature in test_features.items():
    query_features_id.append(image_id)
    query_features.append(feature)

total_queries = len(query_features)
query_features = [tensor.flatten().cpu().detach().numpy() for tensor in query_features]
query_features = np.array(query_features)
query_features = scaler.fit_transform(query_features)
distances, indices = knn.kneighbors(query_features)


correct = 0
for query_index, neighbors_indices in enumerate(indices):
    for indice in neighbors_indices:
        retreived_id = X_train_id[indice]
        query_id = query_features_id[query_index]

        query_labels = test_labels[query_id]
        db_labels = database_labels[retreived_id]
        common_labels = set(query_labels) & set(db_labels)
        if len(common_labels) > 0:
            correct += 1
            print(f"Correct Retrievals: {correct}")
            break
        
evaluation_score = correct / total_queries if total_queries > 0 else 0

print(f"Test Evaluation: {evaluation_score}")



# VISUALIZATION



# Combine training and query features
all_features = np.vstack([X_train, query_features])
all_ids = X_train_id + query_features_id

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
all_features_tsne = tsne.fit_transform(all_features)

# Separate t-SNE transformed features back into training and query
X_train_tsne = all_features_tsne[:len(X_train)]
query_features_tsne = all_features_tsne[len(X_train):]

# Visualize and save the plot
plt.figure(figsize=(10, 8))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], label='Training Data', alpha=0.5)
plt.scatter(query_features_tsne[:, 0], query_features_tsne[:, 1], color='red', label='Query')
# Plot neighbors, etc.

# Save the plot
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of k-NN')
plt.legend()
plt.savefig('/ghome/group01/MCV-C5-G1/Week3/results/task_e/knn_tsne_visualization.png')
plt.close()

# tSNE 2:
X_train_labels = [database_labels[image_id] for image_id in X_train_id]
query_labels = [val_labels[image_id] for image_id in query_features_id]

"""
for i in range(1, 89):

    highlight_label = str(i)

    # Visualize and save the plot
    plt.figure(figsize=(10, 8))

    for i, labels in enumerate(query_labels):
        if highlight_label in labels:
            color = 'red'  # Red color for query features with the highlight label
            alpha = 0.8  # Increase alpha value for red points to make them more visible
        else:
            color = 'blue'  # Blue color for query features without the highlight label
            alpha = 0.2
        plt.scatter(query_features_tsne[i, 0], query_features_tsne[i, 1], color=color, alpha=alpha)

    # Save the plot
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of k-NN')
    plt.savefig(f'/ghome/group01/MCV-C5-G1/Week3/results/task_e/knn_tsne_visualization_query_{highlight_label}.png')
    plt.close()


    # Visualize and save the plot
    plt.figure(figsize=(10, 8))

    # Plot training features with different colors based on the presence of the highlight label
    for i, labels in enumerate(X_train_labels):
        if highlight_label in labels:
            color = 'red'  # Red color for training features with the highlight label
            alpha = 0.8
        else:
            color = 'blue'  # Blue color for training features without the highlight label
            alpha = 0.2
        plt.scatter(X_train_tsne[i, 0], X_train_tsne[i, 1], color=color, alpha=alpha)

    # Save the plot
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of k-NN')
    plt.savefig(f'/ghome/group01/MCV-C5-G1/Week3/results/task_e/knn_tsne_visualization_train_{highlight_label}.png')
    plt.close()
"""
# Create a mapping from COCO class IDs to class names
coco_class_names = {
    '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane',
    '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light',
    '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench',
    '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep',
    '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe',
    '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase',
    '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite',
    '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard',
    '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork',
    '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple',
    '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog',
    '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch',
    '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv',
    '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone',
    '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator',
    '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddy bear',
    '89': 'hair drier', '90': 'toothbrush'
}

# Assign colors to COCO classes
coco_class_colors = {
    'person': 'red', 'bicycle': 'blue', 'car': 'green', 'motorcycle': 'purple', 'airplane': 'orange',
    'bus': 'yellow', 'train': 'cyan', 'truck': 'magenta', 'boat': 'lime', 'traffic light': 'pink',
    'fire hydrant': 'brown', 'stop sign': 'teal', 'parking meter': 'olive', 'bench': 'navy',
    'bird': 'gold', 'cat': 'silver', 'dog': 'crimson', 'horse': 'sienna', 'sheep': 'indigo',
    'cow': 'maroon', 'elephant': 'peru', 'bear': 'tan', 'zebra': 'lavender', 'giraffe': 'orchid',
    'backpack': 'aliceblue', 'umbrella': 'azure', 'handbag': 'beige', 'tie': 'bisque', 'suitcase': 'black',
    'frisbee': 'blanchedalmond', 'skis': 'blueviolet', 'snowboard': 'burlywood', 'sports ball': 'chartreuse',
    'kite': 'chocolate', 'baseball bat': 'coral', 'baseball glove': 'cornflowerblue', 'skateboard': 'cornsilk',
    'surfboard': 'crimson', 'tennis racket': 'cyan', 'bottle': 'darkblue', 'wine glass': 'darkcyan',
    'cup': 'darkgoldenrod', 'fork': 'darkgray', 'knife': 'darkgreen', 'spoon': 'darkkhaki', 'bowl': 'darkmagenta',
    'banana': 'darkolivegreen', 'apple': 'darkorange', 'sandwich': 'darkorchid', 'orange': 'darkred',
    'broccoli': 'darksalmon', 'carrot': 'darkseagreen', 'hot dog': 'darkslateblue', 'pizza': 'darkslategray',
    'donut': 'darkturquoise', 'cake': 'darkviolet', 'chair': 'deeppink', 'couch': 'deepskyblue',
    'potted plant': 'dimgray', 'bed': 'dodgerblue', 'dining table': 'firebrick', 'toilet': 'floralwhite',
    'tv': 'forestgreen', 'laptop': 'fuchsia', 'mouse': 'gainsboro', 'remote': 'ghostwhite', 'keyboard': 'gold',
    'cell phone': 'goldenrod', 'microwave': 'gray', 'oven': 'greenyellow', 'toaster': 'honeydew',
    'sink': 'hotpink', 'refrigerator': 'indianred', 'book': 'ivory', 'clock': 'khaki', 'vase': 'lavenderblush',
    'scissors': 'lawngreen', 'teddy bear': 'lemonchiffon', 'hair drier': 'lightblue', 'toothbrush': 'lightcoral'
}

# Visualize and save the plot
plt.figure(figsize=(10, 8))

# Plot query features with different colors based on the COCO class
for i, labels in enumerate(query_labels):
    for label in labels:
        if label in coco_class_names:
            color = coco_class_colors[coco_class_names[label]]
            plt.scatter(query_features_tsne[i, 0], query_features_tsne[i, 1], color=color, alpha=0.8)


# Save the plot
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of k-NN based on COCO Classes')
plt.legend()
plt.savefig('/ghome/group01/MCV-C5-G1/Week3/results/task_e/a_knn_tsne_visualization_coco_new.png')
plt.close()

