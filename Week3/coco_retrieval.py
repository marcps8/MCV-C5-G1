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
from torch.utils.data import DataLoader
from tqdm import tqdm

from triplets_dataset import TripletsDataset
from utils_week3 import *

TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week3/results/task_e"
ANNOTATIONS_JSON = (
    "/ghome/group01/MCV-C5-G1/Week3/inverted_annotations/inverted_annotations.json"
)
DATA_LOADER = "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/dataloader_new.pkl"
SAVE_PATH = "/ghome/group01/MCV-C5-G1/Week3/results/task_e/models/model_emb128_margin02_p2_epochs2_minibatch_32_8workers.pth"

# faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weigths='COCO_V1')


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(4096, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # print(f'Lo que le llega en nuestro embedding {x["pool"].shape} {x["0"].shape} {x["1"].shape} {x["2"].shape}')
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
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)
triplet_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-7)
model = models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone
embed = EmbeddingLayer(embed_size=128)
model = torch.nn.Sequential(*list(model.children())[:], embed)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)

if os.path.exists(DATA_LOADER):
    with open(DATA_LOADER, "rb") as f:
        triplets = pkl.load(f)

else:
    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    triplets = get_triplets(annotations, "train")
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

training = True
if training:
    model.train()
    num_epochs = 10
    i = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for anchors, positives, negatives in tqdm(train_loader):
            # Move tensors to the correct device
            anchors, positives, negatives = (
                anchors.to(device),
                positives.to(device),
                negatives.to(device),
            )

            optimizer.zero_grad()

            # Forward pass
            anchor_outs = model(anchors)
            pos_outs = model(positives)
            neg_outs = model(negatives)

            # Compute loss
            loss = triplet_loss(anchor_outs, pos_outs, neg_outs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
        )
    torch.save(model.state_dict(), SAVE_PATH)

else:
    model.load_state_dict(torch.load(SAVE_PATH))
    model.to(device)


def extract_features(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transformations(image).unsqueeze(0).to(device).float()

    with torch.no_grad():
        features = model(image_tensor)

    return features


# Extract features from Database to retrieve
if os.path.exists(
    "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features.pkl"
):
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features.pkl", "rb"
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
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/database_features.pkl", "wb"
    ) as f:
        pkl.dump(database_features, f)


# Extract features from Validation
if os.path.exists(
    "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features.pkl"
):
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features.pkl", "rb"
    ) as f:
        validation_features = pkl.load(f)

else:
    validation_features = {}
    for image_id, labels in val_labels.items():
        image_path = os.path.join(
            TRAIN_PATH, "COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
        )
        val_feature = extract_features(model=model, image_path=image_path)
        validation_features[image_id] = val_feature

    os.makedirs("/ghome/group01/MCV-C5-G1/Week3/pickles/task_e", exist_ok=True)
    with open(
        "/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/validation_features.pkl", "wb"
    ) as f:
        pkl.dump(validation_features, f)

# Train NN
with open(
    "/ghome/group01/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json", "r"
) as f:
    annotations = json.load(f)

X_train = []
n_neighbors = len(annotations["database"].keys())
for image_id, feature in database_features.items():
    X_train.append(feature)

X_train = np.array(X_train)
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)

# Evaluate on Validation dataset
query_features = []
for image_id, feature in validation_features.items():
    query_features.append(feature)

total_queries = len(query_features)
query_features = np.array(query_features)
distances, indices = knn.kneighbors(query_features)

correct = 0
for query_index, neighbors_indices in enumerate(indices):
    similar_feature = X_train[neighbors_indices[0]]

    for image_id, feature in database_features.items():

        tolerance = 1e-6
        if np.allclose(similar_feature, feature, atol=tolerance):
            query_id = list(validation_features.keys())(query_index)
            query_labels = validation_features[query_id]
            db_labels = database_labels[image_id]
            common_labels = list(set(query_labels) & set(db_labels))

            if common_labels > 0:
                correct += 1

            break

evaluation_score = correct / total_queries if total_queries > 0 else 0

print(f"Validation Evaluation: {evaluation_score}")
